# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5

    attention_type: Literal["mha", "mla"] = "mha"
    q_lora_rank: int = 512
    kv_lora_rank: int = 512
    rope_head_dim: int = 64
    mla_absorption: bool = True

    # TODO: all the args we need to toggle

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [config for config in transformer_configs if config.lower() in str(name).lower()]

        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(config[1]), name # make sure only one 'best' match
            
        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "CodeLlama-7b-Python-hf": dict(block_size=16384, vocab_size=32000, n_layer=32, dim = 4096, rope_base=1000000),
    "7B": dict(n_layer=32, n_head=32, dim=4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "30B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(n_layer=48, n_head=64, dim=8192, vocab_size=32000, n_local_heads=8, intermediate_size=22016, rope_base=1000000), # CodeLlama-34B-Python-hf
    "70B": dict(n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672),
    # "Mistral-7B": dict(n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=32000),
    "Mistral-7B": dict(block_size=4096, n_layer=32, n_head=32, attention_type="mla", dim=4096, intermediate_size=14336, vocab_size=32000),
    "stories15M": dict(n_layer=6, n_head=6, dim=288),
    "stories110M": dict(n_layer=12, n_head=12, dim=768),

    "llama-3-8b": dict(block_size=8192, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000),
    "llama-3-70b": dict(block_size=8192, n_layer=80, n_head=64, n_local_heads=8, dim=8192, intermediate_size=28672, vocab_size=128256, rope_base=500000),
}

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out

class MLACache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, kv_lora_rank, rope_head_dim, dtype=torch.bfloat16):
        super().__init__()
        compressed_kv_shape = (max_batch_size, max_seq_length, kv_lora_rank)
        k_rope_shape = (max_batch_size, max_seq_length, 1, rope_head_dim)

        self.register_buffer('compressed_kv_cache', torch.zeros(compressed_kv_shape, dtype=dtype))
        self.register_buffer('k_rope_cache', torch.zeros(k_rope_shape, dtype=dtype))

    def update(self, input_pos, compressed_kv_val, k_rope_val):
        # input_pos: [S], compressed_kv_val: [B, S, KV_rank], k_rope_val: [B, S, 1, H_rope]
        assert input_pos.shape[0] == compressed_kv_val.shape[1]

        compressed_kv_out = self.compressed_kv_cache
        k_rope_out = self.k_rope_cache
        compressed_kv_out[:, input_pos, :] = compressed_kv_val
        k_rope_out[:, input_pos, :, :] = k_rope_val

        return compressed_kv_out, k_rope_out



class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size, max_seq_length):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.output.weight.dtype
        # For quantized layers, dtype is encoded in scales
        if hasattr(self.output, "scales"):
            dtype = self.output.scales.dtype
        elif hasattr(self.output, "scales_and_zeros"):
            dtype = self.output.scales_and_zeros.dtype
        for b in self.layers:
            if self.config.attention_type == "mha":
                b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim, dtype)
            else:
                b.attention.kv_cache = MLACache(max_batch_size, max_seq_length, self.config.kv_lora_rank, self.config.rope_head_dim, dtype)

        if self.config.attention_type == "mha":
            self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base, dtype)
        else:
            self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.rope_head_dim, self.config.rope_base, dtype)
        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis
        x = self.tok_embeddings(idx)

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config) if config.attention_type == "mha" else LatentAttention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis[input_pos])
        k = apply_rotary_emb(k, freqs_cis[input_pos])

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        return y


class LatentAttention(nn.Module):
    """Fast Multi-headed Latent Attention. 

    This implementation referenced the published code in 
    https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/modeling_deepseek.py
    and implements some of the optimizations described in the MAD-sys blogpost (although I have not atm referenced the madsys code when writing this.)
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        # TODO: size checks, as with Attention above
        self.n_head = config.n_head

        self.nope_head_dim = config.head_dim
        self.rope_head_dim = config.rope_head_dim # TODO: un-hardcode

        self.q_head_dim = self.nope_head_dim + self.rope_head_dim

        self.v_head_dim = self.nope_head_dim #TODO: can this be different from QK head dim?

        self.dim = config.dim 

        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank

        # init q_lora proj (OR: full-rank wq)
        # as well as RMSNorm if lora
        if self.q_lora_rank is not None:
            self.wdq = nn.Linear(self.dim, self.q_lora_rank, bias=False)
            self.q_lora_norm = RMSNorm(self.q_lora_rank, config.norm_eps)
            self.wuq = nn.Linear(self.q_lora_rank, self.n_head * self.q_head_dim, bias=False)
        else:
            self.wq = nn.Linear(self.dim, self.n_head * self.q_head_dim, bias=False)


        # init kv lora proj
        # (combined W_DKV and W_KR from MLA eqns)
        self.wdkv_kr = nn.Linear(self.dim, self.kv_lora_rank + self.rope_head_dim, bias=False)

        self.kv_lora_norm = RMSNorm(self.kv_lora_rank, config.norm_eps)

        # combined w_uk, w_uv
        self.wukv = nn.Linear(config.kv_lora_rank, self.n_head * (self.nope_head_dim + self.v_head_dim), bias=False)

        self.wo = nn.Linear(self.n_head * self.v_head_dim, self.dim, bias=False)

        self.kv_cache = None

        if config.mla_absorption: 
            self.absorb()

        # TODO: write a load hook to make us not require ckpt conversion script w/ renaming?
        # TODO: load hook should handle the "absorption" effect. if we even wanna do absorption
        # TODO TODO: if absorption isn't "worth it" in terms of speedups in practicality--just stick a RoPE in there!
            # TODO: check if ditching the RoPE head gives us a speedup after we perform move elision. it may not be a slowdown to use the uncompressed rope head.

        # TODO: figure out what's going on w/ expansion ratios in Deepseek MLA (Jianlin Su blogpost: head_dim * n_head = 16384 instead of dim = 5120 for Deepseek-v2. Why/how?)

    def absorb(self):
        assert self.config.q_lora_rank is not None

        # in: WUQ: [n_head * q_head_dim, q_lora_rank], WUKV: [n_head * (q_head_dim), kv_lora_rank], WO: [dim, n_head * v_head_dim]

        # split apart RoPE and NoPE components of the Q up-projection (W_UQ), and store RoPE portion separately        
        wuq = self.wuq.weight.view(self.n_head, self.q_head_dim, self.q_lora_rank)

        wuq_nope, wuq_rope = wuq.split([self.nope_head_dim, self.rope_head_dim], dim=1)

        self.wuq_rope = wuq_rope.clone().contiguous().to(dtype=torch.bfloat16)

        # split K,V up-proj (W_UKV) into K and V components
        wukv = self.wukv.weight.view(self.n_head, -1, self.kv_lora_rank)

        wuk, wuv = wukv.split([self.nope_head_dim, self.v_head_dim], dim=1)

        self.wuq_nope = wuq_nope.clone().contiguous().to(dtype=torch.bfloat16)

        self.wuk = wuk.clone().contiguous().to(dtype=torch.bfloat16)

        self.wuv = wuv.clone().contiguous().to(dtype=torch.bfloat16)

        # get W_O
        wo = self.wo.weight.view(self.dim, self.n_head, self.v_head_dim)

        # produce 'W_UQK', the absorbed combined Q- and K- NoPE up-projection: 
        # wuqk = torch.einsum("n_head nope_head_dim q_lora_rank, n_head nope_head_dim kv_lora_rank -> kv_lora_rank n_head q_lora_rank", wuq_nope, wuk)
        # wuqk = torch.einsum("n h q, n h k -> k n q", wuq_nope, wuk)
        # self.wuqk = wuqk.clone().contiguous().to(dtype=torch.bfloat16)

        # TODO: test this and compare--what if we don't absorb W_UV into W_O?
        self.wuv = wuv.clone().contiguous().to(dtype=torch.bfloat16)
        # produce 'W_OV', the absorbed combined V- up-proj and O out projection
        # wov = torch.einsum("n_head v_head_dim kv_lora_rank, dim n_head v_head_dim -> dim n_head kv_lora_rank", wuv, wo)
        # wov = torch.einsum("n v k, d n v -> d n k", wuv, wo)
        # self.wov = wov.clone().contiguous().to(dtype=torch.bfloat16)

        # out: WUQK: [kv_lora_rank, n_head, q_lora_rank], WUQ_rope: [n_head * rope_head_dim, q_lora_rank], WOV: [dim, n_head, kv_lora_rank]



    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        if self.config.mla_absorption:
            return self.forward_absorbed(x, freqs_cis, mask, input_pos)
        
        bsz, seqlen, _ = x.shape

        # run q proj (Q_DQ, Q_UQ) and get Q_nope, Q_pe
        if self.q_lora_rank is not None:
            q = self.wuq(self.q_lora_norm(self.wdq(x)))
        else:
            q = self.wq(x)

        q = q.view(bsz, seqlen, self.n_head, self.q_head_dim)
        q_nope, q_rope = q.split([self.nope_head_dim, self.rope_head_dim], dim=-1)

        # run combined DKV + KR --> get compressed KV and k_rope
        compressed_kv, k_rope = self.wdkv_kr(x).split([self.kv_lora_rank, self.rope_head_dim], dim=-1)
        # TODO: should KR proj be separate from DKV weight? or better to do it combined
        k_rope = k_rope.view(bsz, -1, 1, self.rope_head_dim)

        # TODO: push kv LN prior to caching?

        # use KV cache: cache the compressed_kv ; uncompressed rotary K
        if self.kv_cache is not None:
            # TODO write a MLA (compressed) KV cache
            compressed_kv, k_rope = self.kv_cache.update(input_pos, compressed_kv, k_rope)

        # up-project compressed_kv and split out uncompressed k_nope and value states
        # run wUK, wUV to get uncompressed k_nope and get uncompressed v
        kv = self.wukv(self.kv_lora_norm(compressed_kv))

        k_nope, v = kv.view(bsz, -1, self.n_head, self.nope_head_dim + self.v_head_dim).split([self.nope_head_dim, self.v_head_dim], dim=-1)

        # apply rotary to q_rope, k_rope
        q_rope = apply_rotary_emb(q_rope, freqs_cis[input_pos])
        k_rope = apply_rotary_emb(k_rope, freqs_cis[:k_rope.shape[1], ...]) # we are reapplying RoPE to the entire k_rope history each time at runtime.

        # TODO: don't use cat -- unless torch compile does a good job of making things fast
        # repeat our RoPE MQA K head.
        k_rope = k_rope.repeat_interleave(self.n_head, dim=2)

        # concat k_nope, k_rope together ; q_nope, q_rope together
        q = torch.cat((q_nope, q_rope), dim=-1)
        k = torch.cat((k_nope, k_rope), dim=-1)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        # do attention computation.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0) # TODO: ensure we are using the correct softmax_scale

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        return y

    def forward_absorbed(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:

        bsz, seqlen, _ = x.shape


        c_q = self.q_lora_norm(self.wdq(x))

        #q_nope = torch.einsum('bsz seqlen q_lora_rank, kv_lora_rank n_head q_lora_rank -> bsz seqlen n_head kv_lora_rank', c_q, self.wuqk)
        q_nope = torch.einsum('b s q, n h q -> b s n h', c_q, self.wuq_nope)

        q_nope = torch.einsum('b s n h, n h k -> b s n k', q_nope, self.wuk)
        # q_rope = torch.einsum('bsz seqlen q_lora_rank, n_head rope_head_dim q_lora_rank -> bsz seqlen n_head rope_head_dim', c_q, self.wuq_rope)
        q_rope = torch.einsum('b s q, n r q -> b s n r', c_q, self.wuq_rope)

        # run combined DKV + KR --> get compressed KV and k_rope
        compressed_kv, k_rope = self.wdkv_kr(x).split([self.kv_lora_rank, self.rope_head_dim], dim=-1)
        # TODO: should KR proj be separate from DKV weight? or better to do it combined
        k_rope = k_rope.view(bsz, -1, 1, self.rope_head_dim)

        # use KV cache: cache the compressed_kv ; uncompressed rotary K
        if self.kv_cache is not None:
            # TODO write a MLA (compressed) KV cache
            compressed_kv, k_rope = self.kv_cache.update(input_pos, compressed_kv, k_rope)

        compressed_kv = self.kv_lora_norm(compressed_kv)

        # # up-project compressed_kv and split out uncompressed k_nope and value states
        # # run wUK, wUV to get uncompressed k_nope and get uncompressed v
        # kv = self.wukv(self.kv_lora_norm(compressed_kv))

        # k_nope, v = kv.view(bsz, -1, self.n_head, self.nope_head_dim + self.v_head_dim).split([self.nope_head_dim, self.v_head_dim], dim=-1)

        # apply rotary to q_rope, k_rope
        q_rope = apply_rotary_emb(q_rope, freqs_cis[input_pos])
        k_rope = apply_rotary_emb(k_rope, freqs_cis[:k_rope.shape[1], ...]) # we are reapplying RoPE to the entire k_rope history each time at runtime.

        # TODO: don't use cat -- unless torch compile does a good job of making things fast
        # repeat our RoPE MQA K head.
        k_rope = k_rope.repeat_interleave(self.n_head, dim=2) # TODO: still want this?

        # print(compressed_kv.shape)
        # concat k_nope, k_rope together ; q_nope, q_rope together
        # q = torch.cat((q_nope, q_rope), dim=-1)
        # k = torch.cat((compressed_kv, k_rope), dim=-1)

        q_rope, k_rope, q_nope = map(lambda x: x.transpose(1, 2), (q_rope, k_rope, q_nope))

        # do attention computation.
        y = my_scaled_dot_product_attention(q_rope, q_nope, k_rope, compressed_kv, attn_mask=mask, dropout_p=0.0, scale=self.q_head_dim)

        # y = torch.einsum('bsz n_head seqlen kv_lora_rank, dim n_head kv_lora_rank -> bsz seqlen dim', y, self.wov)
        y = torch.einsum('b n s k, n v k -> b n s v', y, self.wuv)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        return y


def my_scaled_dot_product_attention(q_rope, q_nope, k_rope, compressed_kv, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    """Copy of the torch SDPA reference code, edited to compute w/o 
    torch.cat or extra copies. Somewhat cursed    
    """
    
    L, S = q_rope.size(-2), k_rope.size(-2)
    assert scale is not None, "require passing scale explicitly"
    assert attn_mask is not None, "assuming we pass attn_mask explicitly"
    scale_factor = 1 / math.sqrt(q_rope.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(1, 1, L, S, dtype=q_rope.dtype, device=q_rope.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(q_rope.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    # as in madsys blogpost, split out rope and nope attn weight computation
    attn_weight = q_rope @ k_rope.transpose(-2, -1) * scale_factor
    # TODO: ensure the unsqueeze is doing what we want here
    attn_weight += q_nope @ compressed_kv.unsqueeze(-3).transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ compressed_kv


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# TODO: impl. DeepSeekMoE moe layer ; make Deepseek-v2-lite loadable in gpt-fast


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000,
    dtype: torch.dtype = torch.bfloat16
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
