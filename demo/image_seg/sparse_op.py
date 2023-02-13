# import cogdl.data
# from cogdl.utils import spmm
import time

import torch

def cluster_sum(a : torch.LongTensor, b : torch.Tensor, clip_length = 0):
    sz = torch.max(a) + 1 if clip_length == 0 else clip_length
    # assert b.shape[0] >= sz
    if len(b.shape) == 1:
        return torch.zeros(sz, dtype=b.dtype, device=b.device).scatter_add_(0, a, b)
    else:
        sp = list(b.shape)
        sp[0] = sz
        return torch.zeros(sp, dtype=b.dtype, device=b.device).scatter_add_(0, a.unsqueeze(-1).expand_as(b), b)

def cluster_cnt(a : torch.LongTensor, dtype = torch.float, clip_length = 0):
    sz = torch.max(a) + 1 if clip_length == 0 else clip_length
    return torch.zeros(sz, dtype = dtype, device=a.device).scatter_add_(0, a, torch.ones(a.shape[0], dtype = dtype, device=a.device))




if __name__ == '__main__':
    dat = [[1,2,4]] * 3
    print(cluster_sum(torch.ones(3).long(), torch.Tensor(dat)))
    print(cluster_sum(torch.ones(3).long(), torch.Tensor(dat[0]), clip_length=3))
    print(cluster_cnt(torch.ones(3).long()))
def test(t):
    torch.cuda.synchronize()
    clock = time.time()
    torch.unique(t, return_inverse=True)
    torch.cuda.synchronize()
    print(time.time() - clock)
