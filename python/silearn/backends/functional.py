from inspect import isfunction

from silearn.graph import Graph

# (f, backend) --> f
__function_map__ = dict()


def get_dat_backend(dat):
    if isinstance(dat, torch.Tensor):
        return "torch"

def vertex_reduce(g: Graph, partition):
    if not __function_map__.__contains__((vertex_reduce, g.backend)):
        raise NotImplementedError(f"node_reduce() is not available for backend {g.backend}")
    return __function_map__[vertex_reduce, g.backend](g, partition)



def scatter_sum(tensor, idx, clip_length=0):
    backend = get_dat_backend(tensor)
    if not __function_map__.__contains__((scatter_sum, backend)):
        raise NotImplementedError(f"scatter_sum() is not available for backend {backend}")
    return __function_map__[scatter_sum, backend](tensor, idx, clip_length)


def scatter_cnt(idx, clip_length=0):
    backend = get_dat_backend(idx)
    if not __function_map__.__contains__((scatter_cnt, backend)):
        raise NotImplementedError(f"scatter_cnt() is not available for backend {backend}")
    return __function_map__[scatter_cnt, backend](idx, clip_length)

def scatter_max(tensor, idx):
    backend = get_dat_backend(tensor)
    if not __function_map__.__contains__((scatter_max, backend)):
        raise NotImplementedError(f"scatter_max() is not available for backend {backend}")
    return __function_map__[scatter_max, backend](tensor, idx)


def uncertainty(p):
    backend = get_dat_backend(p)
    if not __function_map__.__contains__((uncertainty, backend)):
        raise NotImplementedError(f"uncertainty() is not available for backend {backend}")
    return __function_map__[uncertainty, backend](p)



def entropy(p, q):
    backend = get_dat_backend(p)
    if not __function_map__.__contains__((entropy, backend)):
        raise NotImplementedError(f"entropy() is not available for backend {backend}")
    return __function_map__[entropy, backend](p, q)


# TODO
"""
@:param
    backends ∈ {"torch", "numpy"}
"""
def convert_backend(p, backend):
    backend = get_dat_backend(p)
    if not __function_map__.__contains__((convert_backend, backend)):
        raise NotImplementedError(f"convert_backend() is not available for backend {backend}")
    return __function_map__[convert_backend, backend](p, backend)


# def convert_backend(g: Graph, backend):
#     if not __function_map__.__contains__((convert_backend, g.backend)):
#         raise NotImplementedError(f"convert_backend() is not available for backend {g.backend}")
#     return __function_map__[convert_backend, g.backend](g, backend)



# noinspection PyUnresolvedReferences

__all__ = ["vertex_reduce",
           "convert_backend",
           "scatter_sum",
           "scatter_cnt",
           "scatter_max",
           "entropy",
           "uncertainty",
           "convert_backend"]

# from .functional import *

def __include_functions__(lib, name):
    for k, v in lib.__dict__.items():
        try:
            if isfunction(v) and isfunction(eval(k)):
                __function_map__[eval(k), name] = v
        except:
            pass


try:
    import torch
    import silearn.backends.torch_ops as torch_ops
    __include_functions__(torch_ops, "torch")
finally:
    pass

