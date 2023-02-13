import networkx

import silearn
from silearn.utils.data import GraphEncoding, Graph
from silearn import *


class OneDim(GraphEncoding):
    def uncertainty(self, es, et, p):
        v1 = self.graph.stationary_dist[es]
        return uncertainty(v1)




class Partitioning(GraphEncoding):
    node_id = None # :torch.LongTensor

    def __init__(self, g: Graph, par):
        super().__init__(g)
        self.node_id = par

    def uncertainty(self, es, et, p):
        v1e = self.graph.stationary_dist[es]
        id_et = self.node_id[et]
        id_es= self.node_id[es]
        v2 = scatter_sum(self.graph.stationary_dist, self.node_id)
        v2e = v2[id_es]
        flag = id_es != id_et
        # print(v1e, v2, flag)
        return uncertainty(v1e / v2e) + flag * uncertainty(v2e / v2.sum())

    def structural_entropy(self, reduction = "vertex", norm = False):
        e = super(Partitioning, self).structural_entropy(reduction, norm)
        if reduction == "module":
            et = self.graph.edges[2]
            return scatter_sum(e, self.node_id[et])
        return e

    def to_graph(self):
        import numpy as np
        import torch

        a = np.array([[0, 1.2, 0], [2, 3.1, 0], [0.5, 0, 0]])
        idx = a.nonzero()  # (row, col)
        data = a[idx]

        # to torch tensor
        idx_t = torch.LongTensor(np.vstack(idx))
        data_t = torch.FloatTensor(data)
        coo_a = torch.sparse_coo_tensor(idx_t, data_t, a.shape)


    def to_networkx(self, create_using = networkx.DiGraph(), label_name = "partition"):
        nxg = self.graph.to_networkx(create_using=create_using)
        label_np = silearn.convert_backend(self.node_id, "numpy")
        for i in range(label_np.shape[0]):
            nxg._node[i][label_name] = label_np[i]
        return nxg



class EncodingTree(GraphEncoding):
    parent_id: []

    def uncertainty(self, es, et, p):
        v1 = self.graph.stationary_dist[et]
        cur_ids = es
        cur_idt = et
        ret = 0
        for i in range(len(self.parent_id)):
            id_es = self.parent_id[i][cur_ids]
            id_et = self.parent_id[i][cur_idt]
            vp = scatter_sum(v1, id_et)[id_et] if i != len(self.parent_id) - 1 else v1.sum()
            if i == 0:
                ret += uncertainty(v1 / vp)
            else:
                flag = cur_ids != cur_idt
                ret += flag * uncertainty(v1 / vp)
            v1 = vp
            cur_ids, cur_idt = id_es, id_et
        return ret

    def structural_entropy(self, reduction = "vertex", norm = False):
        e = super(EncodingTree, self).structural_entropy(reduction, norm)
        if reduction.startswith("level"):
            level = int(reduction[5:])
            level = min(-len(self.parent_id), level)
            level = max(len(self.parent_id)-1, level)
            et = self.graph.edges[2]
            return scatter_sum(e, self.parent_id[level][et])
        return e