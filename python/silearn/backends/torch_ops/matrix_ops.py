import numpy as np
import torch
import torch_scatter


def scatter_sum(b : torch.Tensor, a : torch.LongTensor, clip_length = 0):
    sz = torch.max(a) + 1 if clip_length == 0 else clip_length
    # assert b.shape[0] >= sz
    if len(b.shape) == 1:
        return torch.zeros(sz, dtype=b.dtype, device=b.device).scatter_add_(0, a, b)
    else:
        sp = list(b.shape)
        sp[0] = sz
        return torch.zeros(sp, dtype=b.dtype, device=b.device).scatter_add_(0, a.unsqueeze(-1).expand_as(b), b)

def scatter_cnt(a : torch.LongTensor, dtype = torch.float, clip_length = 0):
    sz = torch.max(a) + 1 if clip_length == 0 else clip_length
    return torch.zeros(sz, dtype = dtype, device=a.device).scatter_add_(0, a, torch.ones(a.shape[0], dtype = dtype, device=a.device))


eps_dtype = {
    torch.float64: 1e-306,
    torch.double: 1e-306,
    torch.float32: 1e-36,
    torch.bfloat16: 1e-36,
    torch.float16: 1e-7,
}


# p * log_2 q
def entropy(p: torch.Tensor, q:torch.Tensor):
    dtype = p.dtype
    eps = eps_dtype[dtype] if eps_dtype.keys().__contains__(dtype) else 1e-36
    return -p * torch.log2(torch.clip(q, min = eps))


def uncertainty(q: torch.Tensor):
    dtype = q.dtype
    eps = eps_dtype[dtype] if eps_dtype.keys().__contains__(dtype) else 1e-36
    return -torch.log2(torch.clip(q, min = eps))


scatter_max = torch_scatter.scatter_max

import torch.utils.dlpack

try:
    import cupy
except:
    pass

def convert_backend(p : torch.Tensor, backend: str):
    if backend == "numpy":
        # force
        return p.numpy(force = True)
    elif backend == "cupy":
        # https://docs.cupy.dev/en/stable/user_guide/interoperability.html
        return cupy.asarray(p)
    elif backend == "dlpack":
        # noinspection PyUnresolvedReferences
        return torch.utils.dlpack.to_dlpack(p)
    else:
        raise NotImplementedError(f"convert_backend is not implemented for (torch.Tensor, {str(backend)})")

