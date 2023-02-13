
from si_algorithms import TwoDimSE
import ctypes
import os
import numpy as np
import torch
dirs = os.listdir("./cpp/cpp")
try:
    libSI = None
    import platform
    if platform.system() == 'Linux':
        exec=".so"
    else:
        exec=".dll"
    for x in dirs:
        if x.endswith(exec):
            libSI=ctypes.CDLL("./cpp/cpp/"+x)
    native_si = libSI.si
    # native_si = libSI.si_hyper
except Exception as e:
    # pass
    print(e)
    raise Exception("SI lib is not correctly compiled")


# By Hujin
class NativeTwoDimSE(TwoDimSE):
    def process(self, erase_loop=True, adj_cover = None):
        edge_s = self.graph.edge_index[0].cpu().numpy().astype(np.uint32).ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        edge_t = self.graph.edge_index[1].cpu().numpy().astype(np.uint32).ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        edge_w = self.graph.edge_weight.cpu().numpy().astype(np.double).ctypes.data_as(ctypes.POINTER(ctypes.c_double))


        assert self.graph.edge_index[0].shape[0] == self.graph.num_edges and \
               self.graph.edge_index[1].shape[0] == self.graph.num_edges and \
               self.graph.edge_weight.shape[0] == self.graph.num_edges

        assert self.graph.edge_index[0].max() < self.node_cnt and self.graph.edge_index[1].max() < self.node_cnt
        assert self.graph.edge_index[0].min() >= 0 and self.graph.edge_index[1].min() >= 0

        result = np.zeros(self.node_cnt, np.uint32)
        result_pt = result.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))

        if erase_loop:
            if self.target_cluster > 0:
                if adj_cover != None:
                    adj_cover = (adj_cover > 0).cpu().numpy().astype(np.uint32).ctypes.data_as(
                        ctypes.POINTER(ctypes.c_uint32))
                    libSI.si_tgt_adj(self.node_cnt, self.graph.num_edges, edge_s, edge_t, edge_w, result_pt, ctypes.c_double(self.m_scale), adj_cover)
                else:
                    libSI.si_tgt(self.node_cnt, self.graph.num_edges, edge_s, edge_t, edge_w, result_pt, ctypes.c_double(self.m_scale))


            else:
                libSI.si(self.node_cnt, self.graph.num_edges, edge_s, edge_t, edge_w, result_pt)
        elif adj_cover is not None:
            adj_cover = (adj_cover > 0).cpu().numpy().astype(np.uint32).ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
            libSI.si_hyper_adj(self.node_cnt, self.graph.num_edges, edge_s, edge_t,  edge_w, adj_cover,result_pt)
        else:
            libSI.si_hyper(self.node_cnt, self.graph.num_edges, edge_s, edge_t, edge_w, result_pt)


        self.community_result = torch.LongTensor(result.astype(np.int64)).to(self.graph.device)







