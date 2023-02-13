import scipy
import torch
import torch_scatter

import silearn

import networkx



# Graph Model for Random Walk Process
#
class Graph:
    # adj : torch.Tensor
    backend = "torch"
    directed = True

    def __init__(self):
        pass


    @property
    def device(self):
        return "cpu"

    @property
    def num_vertices(self):
        return 0

    @property
    def num_edges(self):
        return 0

    def vertex_reduce(self,  partition):
        silearn.vertex_reduce(self, partition)

    def to_networkx(self, create_using = networkx.DiGraph()):
        raise NotImplementedError("Not Implemented")

    @property
    def stationary_dist(self):
        raise NotImplementedError("Not Implemented")

    @property
    def vertex_weight_es(self):
        raise NotImplementedError("Not Implemented")

    @property
    def edges(self):
        raise NotImplementedError("Not Implemented")

    def query_weight(self, es, et):
        raise NotImplementedError("Not Implemented")



class GraphSparse(Graph):
    _edges: torch.Tensor
    _p: torch.Tensor
    _dist = None
    n_vertices = 0

    def __init__(self, edges, p):
        super().__init__()
        self.n_vertices = int(edges.max())+ 1
        self._edges, self._p = edges, p


    @property
    def device(self):
        return self._edges.device

    @property
    def num_vertices(self):
        return self.n_vertices

    @property
    def num_edges(self):
        return self._edges.shape[0]

    @property
    def vertex_weight_es(self):
        return silearn.scatter_sum(self._p, self.edges[0][:, 0])


    @property
    def edges(self):
        flag = self._p > 0
        return self._edges[flag, :], self._p[flag]

    @property
    def stationary_dist(self):
        return self._dist

    def to_networkx(self, create_using = networkx.DiGraph()):
        edges = silearn.convert_backend(self._edges, "numpy")
        weights = silearn.convert_backend(self._p, "numpy")
        scipy.sparse.coo.coo_matrix((weights, (edges[:, 0], edges[:, 1])), (self.n_vertices, self.n_vertices))
        networkx.from_scipy_sparse_array(edges, create_using=create_using)

    # def query_weight(self, es, et):
    #





# return : one dim tensor
class GraphDense(Graph):
    adj: torch.Tensor

    def to_sparse(self):
        raise NotImplementedError("Not Implemented")

    # todo: move
    def edges(self):
        edges = torch.nonzero(self.adj)
        return edges, self.adj[edges[:, 0]][edges[:, 1]]

    def query_weight(self, es, et):
        return self.adj[es][et]





class GraphEncoding:
    def __init__(self, g : Graph):
        self.graph = g

    def uncertainty(self, es, et, p):
        raise NotImplementedError("Not Implemented")

    def positioning_entropy(self):
        dist = self.graph.stationary_dist
        return silearn.entropy(dist, dist)

    def entropy_rate(self, reduction = "vertex", norm = False):
        e, p = self.graph.edges
        es, et = e[:, 0], e[:, 1]
        nw = self.graph.vertex_weight_es[es]
        e = silearn.entropy(p, p / nw)

        if norm:
            dist = self.graph.stationary_dist[es]
            e = e / silearn.entropy(dist, dist)

        if reduction == "none":
            return e
        elif reduction == "vertex":
            return silearn.scatter_sum(e, et)
        elif reduction == "sum":
            return e.sum()
        else:
            return e

    def structural_entropy(self, reduction = "vertex", norm = False):
        e, p = self.graph.edges
        es, et = e[:, 0], e[:, 1]
        # dist = self.graph.stationary_dist[es]
        dist = self.graph.stationary_dist[es]
        # tot = w.sum()
        e = p * self.uncertainty(es, et, p)

        if norm:
            e = e / silearn.entropy(dist, dist)
        if reduction == "none":
            return e
        elif reduction == "vertex":
            return silearn.scatter_sum(e, et)
        elif reduction == "sum":
            return e.sum()
        else:
            return e

    def to_networkx(self, create_using = networkx.DiGraph()):
        raise NotImplementedError()


