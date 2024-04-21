from dejavu_kernels import mlp_sparse
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from triton.testing import do_bench as triton_bench

from dataclasses import dataclass

@dataclass
class ModelArgs:
    dim = 4096
    intermediate_size = 14336
    ff_sparsity = 256
    controller_rank = 16

class ReluFeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.relu(self.w1(x)))



ffn = ReluFeedForward(ModelArgs()).to("cuda").to(dtype=torch.bfloat16).eval()

class SparseController(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.c1 = nn.Linear(config.dim, config.controller_rank, bias=False)
        self.c2 = nn.Linear(config.controller_rank, config.intermediate_size, bias=False)

        self.sparsity = config.ff_sparsity
        self.d_model = config.intermediate_size

    def forward(self, x: Tensor) -> Tensor:
        BT, _ = x.size() # (T = seqlen)
        device = x.device
        # xC_{1}C_{2}
        intermediate = self.c2(self.c1(x))
        # "Reshape (-1, N)" from paper
        # intermediate: [B * T, (d_ff / N), N]
        # out: [B * T, (d_ff / N)] ?
        return torch.argmax(torch.reshape(intermediate, (-1, self.sparsity)), dim=-1) + torch.arange(start=0, end=self.d_model, step=self.sparsity, device=device)

class SparseFeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.controller = SparseController(config)

        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

        self.BACKEND = "sparse"
        self.w2t = self.w2.weight.clone().detach().t().contiguous().to(torch.bfloat16).to("cuda") # ideally we'd transpose + make contiguous W2 ahead of time...

    def forward(self, x: Tensor) -> Tensor:
        B, T, M = x.size() # T = seqlen, M = d_model

        if self.BACKEND == "naive":
            # TODO: convert mask to one-hot in naive backend? this should be naive, indexing in native pytorch
            raise NotImplementedError("only kernel supported right now")
        elif (T > 1):
            #print("prefilling... will perform dense computation")
            import time
            t0 = time.perf_counter()
            out = self.w2(F.relu(self.w1(x)))
            print(f"Prefill time: {time.perf_counter() - t0}")
            return out
        else:
            x = torch.reshape(x, (B * T, M)).contiguous() # x: [B * T, M]
            mask = self.controller(x)

            # if self.w2t is None:
            #     self.w2t = self.w2.weight.t().contiguous()
            # hardcoded to ReLU + no biases
            # TODO: idx should be union of selected rows/cols across B and T axes
            # print(x.shape, self.w1.weight.shape, self.w2t.shape, mask[0, :].shape)
            # print(x.shape, x.dtype, self.w1.weight.shape, self.w2t.shape, mask.shape)
            import time
            t0 = time.perf_counter()
            out = mlp_sparse(x, W1=self.w1.weight, W2t=self.w2t, idx=mask) 
            #out = self.w2(F.relu(self.w1(x)))
            print(f"Sparse time: {time.perf_counter() - t0}")
            # TODO: implement naive weight indexing a la https://github.com/google/trax/blob/a6a508e898a69fecbcce8e5b991666632c629cb0/trax/layers/research/sparsity.py#L1351
            return torch.reshape(out, (B, T, M)).contiguous()

print(ffn)

import functools
do_bench = functools.partial(triton_bench, warmup=0, rep=1)

inp = torch.rand((1,16,4096), dtype=torch.bfloat16, device="cuda")
print(do_bench(lambda: ffn(inp)))

# w2t = ffn.w2.weight.t().contiguous()
sparsity = 256

sparse_ffn = SparseFeedForward(ModelArgs()).to(dtype=torch.bfloat16, device="cuda").eval()
# print(do_bench(lambda: mlp_sparse(inp[0, ...], ffn.w1.weight, w2t, idx=torch.randint(low=0, high=sparsity - 1, size=(14336 // sparsity, ), device="cuda") + torch.arange(0, 14336, step=sparsity, device="cuda"))))
print(do_bench(lambda: sparse_ffn(inp)))

inp = torch.rand((1,1,4096), dtype=torch.bfloat16, device="cuda")
print(do_bench(lambda: ffn(inp)))

# w2t = ffn.w2.weight.t().contiguous()
sparsity = 256

# sparse_ffn = SparseFeedForward(ModelArgs()).to(dtype=torch.bfloat16, device="cuda").eval()
# print(do_bench(lambda: mlp_sparse(inp[0, ...], ffn.w1.weight, w2t, idx=torch.randint(low=0, high=sparsity - 1, size=(14336 // sparsity, ), device="cuda") + torch.arange(0, 14336, step=sparsity, device="cuda"))))
print(do_bench(lambda: sparse_ffn(inp)))
