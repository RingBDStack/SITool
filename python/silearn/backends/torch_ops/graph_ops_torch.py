import torch


from silearn.graph import GraphSparse, GraphDense


def vertex_reduce_sparse(g:GraphSparse):
    pass


def nonzero_idx_dense(g: GraphDense, return_weights = True):
    es, et = torch.nonzero(g.adj, as_tuple=True)
    if return_weights:
        return es, et, g.adj[es][et]
    else:
        return es, et

