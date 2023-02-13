import heapq
import time

import cogdl.utils
import math
import numpy as np
import torch
import torch_scatter
from cogdl.data import Graph
from cogdl.utils import spmm
from numpy import log2

from sparse_op import cluster_sum, cluster_cnt

# By Hujin
class TwoDimSE:
    _eps = 1e-15
    target_cluster = -1
    erase_loop = False

    def __init__(self, graph: Graph):
        self.graph = graph
        self.device = graph.edge_weight.device
        self.node_cnt = int(self.graph.num_nodes)
        self.sum_degree2 = float(graph.edge_weight.sum())

        # results
        self.community_result = None

        self.transform_g = None
        self.bridge_flag = None
        self.bridge_flag_adj = None
        self.community = None
        self.m_scale = 0

    # Build Optimal partition on current graph
    def process(self, erase_loop=True, adj_cover = None):
        self.sum_degree2 = float(self.graph.edge_weight.sum())
        self.node_cnt = int(self.graph.num_nodes)
        self.erase_loop = erase_loop
        graph = self.graph
        vol = np.zeros(self.node_cnt)  # torch.zeros for learning
        g = np.zeros(self.node_cnt)  # torch.zeros for learning
        graph_weight = graph.edge_weight.detach().cpu().numpy()
        graph_edge_index = [graph.edge_index[0].cpu().numpy(), graph.edge_index[1].cpu().numpy()]
        self._community_adj = [[] for _ in range(0, self.node_cnt * 2)]
        for i in range(len(graph_weight)):
            s = int(graph_edge_index[0][i])
            t = int(graph_edge_index[1][i])
            vol[s] += graph_weight[i]
            if s != t:
                g[s] += graph_weight[i]
            # noinspection PyTypeChecker
            self._community_adj[s].append(t)

        for i in range(self.node_cnt):
            self._community_adj[i] = set(self._community_adj[i])
        self._community_vol = vol.tolist() + [0] * self.node_cnt
        self._community_g0 = g.tolist() + [0] * self.node_cnt
        self._community_g = g.tolist() + [0] * self.node_cnt
        self.entropy1 = float(self.calc_e1())
        self.entropy2 = self.entropy1

        self._community_edge = {}
        self._community_pos = self.node_cnt
        self._merge_history = [-1] * (self.node_cnt * 2)
        self._log2m = np.log2(self.sum_degree2) + self.m_scale
        self._father = list(range(0, self.node_cnt * 2))
        self._cluster_count = self.node_cnt

        # self._process_queue = PriorityQueue()
        self._process_queue = []

        for i in range(len(graph_weight)):
            x, y = graph_edge_index[0][i], graph_edge_index[1][i]
            if x < y:
                if (x, y) not in self._community_edge.keys():
                    self._community_edge[x, y] = float(graph_weight[i])
                    self._community_edge[y, x] = float(graph_weight[i])
                else:
                    self._community_edge[x, y] += float(graph_weight[i])
                    self._community_edge[y, x] += float(graph_weight[i])
        for i in range(0, self.node_cnt):
            for x in self._community_adj[i]:
                dh = self._get_dH(i, x)
                if (dh > 0 or self.target_cluster > 0) and i < x:
                    heapq.heappush(self._process_queue, (-dh, [i, x]))

        while len(self._process_queue) > 0:
            f = heapq.heappop(self._process_queue)
            # Recorded node has been merged.
            if self._father[f[1][0]] != f[1][0] or self._father[f[1][1]] != f[1][1]:
                continue
            if self.target_cluster == self._cluster_count:
                break
            self._cluster_count -= 1
            self._merge(f[1][0], f[1][1], -f[0])

        com = dict()
        com_pos = 0
        self.community_result = np.zeros(self.node_cnt)
        for x in range(self.node_cnt):
            f = self._getf(x)
            if f in com.keys():
                self.community_result[x] = com[f]
            else:
                self.community_result[x] = com[f] = com_pos
                com_pos += 1
        self.community_result = torch.LongTensor(self.community_result).requires_grad_(False).to(self.graph.device)
        self._community_adj = self._community_edge = self._merge_history = self._process_queue \
            = self._community_pos = self._community_vol = self._community_g = None

    # 2m * dH = (vi-gi) log vi + (vj-gj) log vj - (vx - gx) log vx + 2 * edge<ij> log 2m
    # gx = gi + gj - 2 * edge<ij>, vx = vi + vj
    def _get_dH(self, i, j, ):

        def h2(v, g, g0):
            return ((g0 - g) * log2(v) if g0 != g else 0) if not self.erase_loop else ((v - g) * log2(v) if v != g else 0)

        vi, vj = self._community_vol[i], self._community_vol[j]
        gi, gj = self._community_g[i], self._community_g[j]
        gi0, gj0 = self._community_g0[i], self._community_g0[j]
        vx = vi + vj
        edge = self._get_edgexy(i, j)
        gx = gi + gj - 2 * edge
        dH = (h2(vi, gi, gi0) + h2(vj, gj, gj0) - h2(vx, gx, gi0 + gj0) +
              2 * edge * self._log2m) / self.sum_degree2
        return dH

    def _get_vg(self, i, j):
        vi, vj = self._community_vol[i], self._community_vol[j]
        gi, gj = self._community_g[i], self._community_g[j]
        vx = vi + vj
        edge = self._get_edgexy(i, j)
        gx = gi + gj - 2 * edge
        return gx, vx, self._community_g0[i]+ self._community_g0[j]

    # edge<ax> = edge<ai> + edge<aj> (memorized seacdrch)
    # y will be updated
    def _get_edgexy(self, x, y):
        if (x, y) in self._community_edge.keys():
            return self._community_edge[x, y]

        elif y >= self.node_cnt and self._merge_history[y] != -1:
            i, j = self._merge_history[y]
            ans = self._get_edgexy(x, i) + self._get_edgexy(x, j)
            if ans > 0 :
                self._community_edge[x, y] = ans
                self._community_edge[y, x] = ans
            return ans
        return 0

    def _merge(self, i, j, dH):
        # print("merge:{} {}, dH = {}".format(i, j, dH))
        p = self._community_pos

        self._father[i] = self._father[j] = p
        self._community_pos += 1
        gx, vx, g0 = self._get_vg(i, j)
        self.entropy2 -= dH
        self._community_vol[p] = vx
        self._community_g[p] = gx
        self._community_g0[p] = g0
        self._merge_history[p] = [i, j]

        # merge adjacency
        # adj = set(self._getf(x) for x in self._community_adj[i])
        # adj.update(self._getf(x) for x in self._community_adj[j])
        # adj.remove(p)
        # self._community_adj[p] = adj
        # self._community_adj[i] = self._community_adj[j] = []

        if len(self._community_adj[i]) > len(self._community_adj[j]):
            self._community_adj[i].update(self._community_adj[j])
            adj = self._community_adj[i]
            self._community_adj[p] = self._community_adj[i]
        else:
            self._community_adj[j].update(self._community_adj[i])
            adj = self._community_adj[j]
            self._community_adj[p] = adj
        adj.discard(i)
        adj.discard(j)

        # Here is O(V ^ 2)
        # Broadcast changes & push into queue
        for x in adj:
            self._community_adj[x].discard(i)
            self._community_adj[x].discard(j)
            self._community_adj[x].add(p)
            dh = self._get_dH(x, p)
            if dh > 0 or self.target_cluster > 0:
                heapq.heappush(self._process_queue, (-dh, [p, x]))
        self._merge_history[p] = -1
        return

    def _getf(self, i):
        if self._father[i] == i:
            return i
        self._father[i] = self._getf(self._father[i])
        return self._father[i]

    def _calc_transform_graph(self):
        if self.transform_g is not None:
            return self.transform_g
        edge_index = [self.community_result, torch.arange(self.node_cnt, dtype=torch.int64, device=self.graph.device).requires_grad_(False)]
        self.transform_g = cogdl.data.Graph(edge_index=edge_index)
        return self.transform_g

    def _calc_community_tensor(self):
        comm = [[] for _ in range(self.node_cnt * 2)]
        for x in range(self.node_cnt):
            # noinspection PyTypeChecker
            comm[self._getf(x)].append(x)
        self.community = comm
        return self.community


    def calc_e1(self, loop_community = False, reduction=True, mmm = None):
        # ones = torch.ones(self.node_cnt, 1,dtype=self.graph.edge_weight.dtype, device=self.device)

        v1 =  cluster_sum(self.graph.edge_index[0], self.graph.edge_weight).reshape(-1)
        double_m = self.graph.edge_weight.sum()
        v1 = torch.clip(v1, self._eps)
        if loop_community:
            with self.graph.local_graph():
                self.graph.edge_weight = self.graph.edge_weight * (self.graph.edge_index[0] != self.graph.edge_index[1])
                g1 = cluster_sum(self.graph.edge_index[0], self.graph.edge_weight).reshape(-1)
        else:
            g1 = v1
        g1s = g1.sum()
        if reduction:
            return -((g1 / double_m) * torch.log2(v1/ ( double_m if mmm is None else mmm * double_m))).sum()
        else:
            return -((g1 / double_m) * torch.log2(v1/ ( double_m if mmm is None else mmm * double_m)))

    def calc_eh(self):
        v1 = cluster_sum(self.community_result[self.graph.edge_index[0]], self.graph.edge_weight).reshape(-1)
        bridge_flag = self.community_result[self.graph.edge_index[0]] != self.community_result[self.graph.edge_index[1]]
        g1 = cluster_sum(self.community_result[self.graph.edge_index[0]], self.graph.edge_weight*bridge_flag).reshape(-1)
        sw = self.graph.edge_weight.sum()
        return -(g1 / sw * torch.log2(torch.clip(v1, 1e-10) / sw)).sum()
    # Calculate H2 on current graph using saved partition
    # New weight will be used if graph weight has been updated
    def calc_entropy(self, loop_community = False, reduction = True, mmm = None, log_M = None, incremental = False):
        # H2
        # ones = torch.ones(self.node_cnt, 1, device=self.device)
        v1 = cluster_sum(self.graph.edge_index[0], self.graph.edge_weight).reshape(-1)
        v2 = cluster_sum(self.community_result, v1)
        if loop_community:
            with self.graph.local_graph():
                self.graph.edge_weight = self.graph.edge_weight * (self.graph.edge_index[0] != self.graph.edge_index[1])
                g1 = cluster_sum(self.graph.edge_index[0], self.graph.edge_weight).reshape(-1)
        else:
            g1 = v1

        # if loop_community:
        self.bridge_flag = self.community_result[self.graph.edge_index[0]] != self.community_result[
        self.graph.edge_index[1]]
        # else:
        #     self.bridge_flag = torch.logical_or(self.community_result[self.graph.edge_index[0]] != self.community_result[
        #     self.graph.edge_index[1]], self.graph.edge_index[0] ==self.graph.edge_index[1])
        with self.graph.local_graph():
            self.graph.edge_weight = self.graph.edge_weight * self.bridge_flag
            g2_ = cluster_sum(self.graph.edge_index[0], self.graph.edge_weight).reshape(-1)
            g2 = cluster_sum(self.community_result, g2_)

        double_m = self.graph.edge_weight.sum()
        g1s = g1.sum()

        v1 = torch.clip(v1, self._eps)
        v2 = torch.clip(v2, self._eps)
        if mmm is not None:
            log_M = math.log2(mmm)
        if reduction:
            e2 = -(g1 / double_m * torch.log2(v1 / v2[self.community_result])).sum()
            e2 -= (g2 / double_m * torch.log2(v2 / double_m)).sum()
            if log_M is not None:
                e2 += (g2 / double_m).sum() * log_M
        else:
            e2 = -(g1 / double_m * torch.log2(v1 / v2[self.community_result]))
            e2 -= (g2_ / double_m * torch.log2(v2[self.community_result] / double_m))
            if log_M is not None:
                e2 += (g2 / double_m) * log_M
        if incremental:
            return e2, ( (g2 / double_m).sum() if reduction else g2 / double_m )
        return e2




    # Calculate Structural Entropy of adjacent matrix 'mat' using saved partition
    # Requires the shape of 'mat' to be matched with processed graph
    def calc_entropy_cross(self, mat: torch.Tensor):
        # H2
        v1 = mat.sum(dim=1)
        v2 = cluster_sum(self.community_result, v1)

        if self.bridge_flag_adj is None:
            x = self.community_result[torch.arange(0, self.node_cnt, device=self.device)].reshape(1, -1).repeat(self.node_cnt, 1)
            self.bridge_flag_adj = x.t() != x
        g2 = cluster_sum(self.community_result, (mat * self.bridge_flag_adj).sum(dim=1))

        double_m = mat.sum()

        e2 = -(v1 / double_m * torch.log2(
            torch.clip(v1, self._eps) / torch.clip(v2[self.community_result], self._eps))).sum()
        e2 -= (g2 / double_m * torch.log2(torch.clip(v2, self._eps) / double_m)).sum()
        return e2

    def calc_entropy_hierarchical(self, segs: list, reduction = True):
        segs.insert(0, torch.arange(self.node_cnt, device=segs[0].device))
        segs.append(torch.zeros((self.node_cnt), dtype=torch.int64, device=segs[0].device))
        vol_cur = cluster_sum(self.graph.edge_index[1], self.graph.edge_weight).reshape(-1)
        v1 = vol_cur
        volG = v1.sum()
        e_tot = None
        for i in range(len(segs) - 1):
            vol_nxt = cluster_sum(segs[i+1], v1)
            self.bridge_flag = segs[i][self.graph.edge_index[0]] != segs[i][self.graph.edge_index[1]]
            g2 = cluster_sum(self.graph.edge_index[1], self.graph.edge_weight * self.bridge_flag).reshape(-1)
            # print(vol_nxt[segs[i+1]]/vol_cur[segs[i]])
            e = -g2 / volG * torch.log2(torch.clip(vol_cur[segs[i]], self._eps) / torch.clip(vol_nxt[segs[i + 1]], self._eps))
            if reduction:
                e = e.sum()
            e_tot = e if e_tot is None else e_tot + e
            # print(float(e))
            vol_cur = vol_nxt
        return e_tot


    def calc_entropy1_cross(self, mat: torch.Tensor):
        v1 = mat.sum(dim=1).reshape(-1, 1)
        double_m = mat.sum()
        return -(v1 / double_m * torch.log2(v1 / double_m)).sum()

    def dec_info(self, *args, **kargs):
        return self.calc_e1(*args, **kargs) - self.calc_entropy(*args, **kargs)

    def norm_dec_info(self, *args, **kargs):
        e1 = self.calc_e1(*args, **kargs)
        return (e1 - self.calc_entropy(*args, **kargs)) / e1
    

    def save_result(self, path=""):
        torch.save(self.community_result, path)

    def load_result(self, path=""):
        self.community_result = torch.load(path, map_location=self.graph.device)

    def force_coding(self, community_result, refine = False):
        self.community_result = community_result
        if refine:
            vis = [False]* self.node_cnt
            pos = int(torch.max(community_result)) + 1
            com_used = [False]* pos
            com = community_result.cpu().numpy()
            g = [[] for i in range(self.node_cnt)]
            e0, e1 = self.graph.edge_index[0], self.graph.edge_index[1]
            e0, e1 = e0.cpu().numpy(), e1.cpu().numpy()
            for u, v in zip(e0, e1):
                g[u].append(v)

            s = []
            for i in range(self.node_cnt):
                if vis[i]:
                    continue
                c0 = com[i]
                if com_used[c0]:
                    c = pos
                    pos += 1
                else:
                    c = c0
                com_used[c0] = True
                s.append(i)
                while len(s) > 0:
                    u = s.pop()
                    if vis[u]:
                        continue
                    com[u] = c
                    vis[u] = True
                    for x in g[u]:
                        if not vis[x] and com[x] == c0:
                            s.append(x)


            self.community_result = torch.Tensor(com).to(self.community_result.device).long()

    def generate_hyper_graph(self,
                             reset_community_id=True,
                             feature_aggregation=None,
                             erase_self_loop=False,
                             edges_x=None,
                             self_loop_w_only = False,
                             edge_duplicate_sgn = None):
        # transfer nodes
        edge_index2 = (self.community_result[self.graph.edge_index[0]], self.community_result[self.graph.edge_index[1]])
        edge_w = self.graph.edge_weight
        max_id = torch.max(self.community_result).long()
        trans = None
        if reset_community_id:
            trans = torch.zeros(max_id + 1, dtype=torch.int64, device=max_id.device)
            mask = (torch.bincount(self.community_result)) > 0
            max_id = mask.sum() - 1
            trans[mask] = torch.arange(max_id + 1, dtype=torch.int64, device=max_id.device)
            edge_index2 = (trans[edge_index2[0]], trans[edge_index2[1]])

        #merge_feature
        x = None
        if self.graph.x != None:
            if feature_aggregation == 'sum':
                c = self.community_result
                if trans is not None:
                    c = trans[c]
                x = cluster_sum(c, self.graph.x, clip_length=max_id + 1)
            elif feature_aggregation == 'mean':
                c = self.community_result
                if trans is not None:
                    c = trans[c]
                x = cluster_sum(c, self.graph.x, clip_length=max_id + 1) / cluster_cnt(c, clip_length=max_id + 1).unsqueeze(1)
            else:
                x = torch.zeros(max_id + 1, device=max_id.device)
        w0 = None
        if erase_self_loop or self_loop_w_only:
            idx = edge_index2[0] != edge_index2[1]
            if self_loop_w_only:
                idw = torch.logical_not(idx)
                w0 = cluster_sum(edge_index2[0][idw],edge_w[idw], clip_length=max_id + 1)
            edge_w = edge_w[idx]
            if edges_x is not None:
                edges_x = edges_x[idx]
            edge_index2 = edge_index2[0][idx], edge_index2[1][idx]
            if edge_duplicate_sgn is not None:
                edge_duplicate_sgn = edge_duplicate_sgn[idx]
        if len(edge_w) == 0:
            return cogdl.data.Graph(edge_index = (torch.LongTensor(), torch.LongTensor()), edge_weight = torch.LongTensor(), x = x)
        # calc edge merging mat

        if edge_duplicate_sgn is not None:
            not_dup = torch.logical_not(edge_duplicate_sgn)
            edge_index2_side = (edge_index2[0][not_dup],
                                edge_index2[1][not_dup])
            edge_index2 = (edge_index2[0][edge_duplicate_sgn],
                            edge_index2[1][edge_duplicate_sgn])

            edges_w_side = edge_w[not_dup]
            edge_w = edge_w[edge_duplicate_sgn]
            if edges_x is not None:
                edges_x_side = edges_x[not_dup]
                edges_x = edges_x[edge_duplicate_sgn]

        edge_hash = edge_index2[0] * (max_id+1) + edge_index2[1]
        trans = torch.unique(edge_hash, return_inverse=True)[1]
        del edge_hash

        # merge weight
        cnt_e = trans.max() + 1
        edge_weight_2 = cluster_sum(trans, edge_w)

        if edges_x is not None:
            edges_x = cluster_sum(trans, edges_x)
        del edge_w
        # merge nodes
        # out[trans[i]]=in[i]
        out = torch.zeros_like(edge_index2[0][:cnt_e]), torch.zeros_like(edge_index2[1][:cnt_e])
        edge_index2 = (out[0].scatter(0, trans, edge_index2[0]), out[1].scatter(0, trans, edge_index2[1]))
        del trans

        if edge_duplicate_sgn is not None:
            edge_index2 = (torch.cat([edge_index2_side[0], edge_index2[0]]),
                           torch.cat([edge_index2_side[1], edge_index2[1]]))
            edge_weight_2 = torch.cat([edges_w_side, edge_weight_2])
            # assert edge_index2[0].shape[0] == edge_weight_2.shape[0]
            if edges_x is not None:
                edges_x = torch.cat([edges_x_side, edges_x])

        if self_loop_w_only:
            idx = torch.arange(max_id+1, device=max_id.device)
            edge_index2 = (torch.cat([idx, edge_index2[0]]), torch.cat([idx, edge_index2[1]]))
            edge_weight_2 = torch.cat([w0, edge_weight_2])
            if edges_x is not None:
                sp = [w0.shape[0]]
                for i in range(1, len(edges_x.shape)):
                    sp.append(edges_x.shape[i])
                edges_x = torch.cat([torch.zeros(sp, dtype=edges_x.dtype, device=edges_x.device), edges_x])

        if edges_x is not None:
            return cogdl.data.Graph(edge_index = edge_index2, edge_weight = edge_weight_2, x=x), edges_x
        return cogdl.data.Graph(edge_index = edge_index2, edge_weight = edge_weight_2, x = x)

    # def merge_fast(self, bd):
    #     v1 = cluster_sum(self.graph.edge_index[1], self.graph.edge_weight).reshape(-1)
    #
    #
    #     flag_min = (self.graph.edge_index[0] < self.graph.edge_index[1])
    #     edge_hash = torch.zeros_like(self.graph.edge_index[0])
    #     edge_hash[flag_min] = self.graph.edge_index[0] * (self.node_cnt+1) + self.graph.edge_index[1]
    #     edge_hash[torch.logical_not(flag_min)] = self.graph.edge_index[1] * (self.node_cnt+1) + self.graph.edge_index[0]
    #     trans = torch.unique(edge_hash, return_inverse=True)[1]
    #     edge_weight2 = cluster_sum(trans,self.graph.edge_weight)
    #
    #     vs = v1[self.graph.edge_index[0]]
    #     vt = v1[self.graph.edge_index[1]]
    #     if torch.any(self.graph.edge_index[0] == self.graph.edge_index[1]):
    #         g1 = cluster_sum(self.graph.edge_index[0],
    #                          self.graph.edge_weight * (self.graph.edge_index[0] != self.graph.edge_index[1])).reshape(-1)
    #         gs = g1[self.graph.edge_index[0]]
    #         gt = g1[self.graph.edge_index[0]]
    #     else:
    #         gs = vs
    #         gt = vt
    #     vx = vs + vt
    #     ee = edge_weight2[trans]
    #     gx = gs + gt - 2 * ee
    #     def h2(v, g):
    #         return (v - g) * torch.log2(v)
    #
    #     dH = h2(vs, gs) + h2(vt, gt) - h2(vx, gx) + 2 * ee * self._log2m
    #     srt = torch.argsort(dH[dH < 0])
    #     srt = srt[:int(srt.shape[0] * bd)]
    #     ids = self.graph.edge_index[0][srt]
    #     idt = self.graph.edge_index[1][srt]


    def process_fast(self, p = 1.0, directed = False, ter = 0, loop0 = None, adj_cover = None, min_com = 1,
                     multiscale=False, max_com = math.inf, di_max=False, srt_M = False,
                     f_hyper_edge = None):
        assert 1 <= min_com <= max_com
        assert 0.0 <= p <= 1.0
        graph_old = self.graph
        graph_old_x = self.graph.x
        self._log2m = torch.log2(self.graph.edge_weight.sum())
        if self.m_scale != 0:
            self._log2m += self.m_scale
        # print(self._log2m)
        self.graph.x = None
        operated_cnt = math.inf
        com0 =  torch.arange(self.graph.num_nodes, device=self.device)
        transs = []
        merge_all = False
        if loop0 is None:
            loop0 = bool((self.graph.edge_index[0] == self.graph.edge_index[1]).any())

        # v1 = None
        vs = None
        vt = None
        dH0 = None

        cache = False




        t = time.time()
        while operated_cnt > ter or not merge_all:
            if operated_cnt <= ter + 1:
                merge_all = True
            # print(self.graph.edge_weight.shape)
            # print(self.graph.edge_index[1].shape)
            # print("time-x:{}".format(time.time() - t))
            # t = time.time()
            # print(self.graph.num_nodes)
            if not cache:
                v1 = cluster_sum(self.graph.edge_index[1], self.graph.edge_weight).reshape(-1)
                vs = v1[self.graph.edge_index[0]]
                vt = v1[self.graph.edge_index[1]]

            if loop0:
                if not cache:
                    g1 = cluster_sum(self.graph.edge_index[1], self.graph.edge_weight * (self.graph.edge_index[0] != self.graph.edge_index[1])).reshape(-1)
                    gs = g1[self.graph.edge_index[0]]
                    gt = g1[self.graph.edge_index[1]]
                    vx = vs + vt
                    gx = gs + gt

                    # g0s, g0t, g0x = vs, vt, vx
                    # if hyper_g:
                    #     if self.graph.x is None:
                    #         self.graph.x = g1.clone()
                    #     else:
                    #         g0 = self.graph.x
                    #         print(g0.shape)
                    #         g0s = g0[self.graph.edge_index[0]]
                    #         g0t = g0[self.graph.edge_index[1]]
                    #         g0x = g0s + g0t
                    dH1 = (vs - gs) * torch.log2(vs) + (vt - gt) * torch.log2(vt) - (vx - gx) * torch.log2(vx)
                    dH2 = 2 * self.graph.edge_weight * ((self._log2m) - torch.log2(vx))
                    dH0 = dH1 + dH2
                    op = (dH0 > 0)
                    # if srt_M:
                    #     dH = - dH2 / dH1 # * (self._log2m)
                        # print(dH1.min())
                        # print(dH2.min())
                        # print(dH.min())
                        # dH = dH0
                    # else:
                    dH = dH0
                    dHM = dH

                else:
                    op = (dH0 > 0)
                    dH = dH0


                if not torch.any(op):
                    break
                if not merge_all:
                    op = (dH >= torch.median(dH[op]))
            else:
                dH = self.graph.edge_weight * (self._log2m - torch.log2(vs + vt))
                op = (dH >= torch.median(dH[dH > 0]))

            cache = True




            if adj_cover is not None:
                op = torch.logical_and(op, adj_cover > 0)


            # print(time.time() - t)



            # print(operated_cnt)
            merge = op
            # rand_idx = torch.randint(0,  2**31,(1, self.graph.num_nodes), device=self.device)[0]
            # noinspection PyTypeChecker
            # merge = torch.logical_and(merge, (vs < vt) + torch.logical_and((vs == vt) , rand_idx[self.graph.edge_index[0]] < rand_idx[self.graph.edge_index[1]]))
            # merge = torch.logical_and(merge, (vs < vt) + torch.logical_and((vs == vt) , rand_idx[self.graph.edge_index[0]] < rand_idx[self.graph.edge_index[1]]))
            merge = torch.logical_and(merge,
                                      torch.logical_or((vs < vt) ,
                                                              torch.logical_and((vs == vt) ,
                                                                                self.graph.edge_index[0]  < self.graph.edge_index[1])))
            if not torch.any(merge):
                operated_cnt = 0
                continue

            # t = time.time()
            id0 = self.graph.edge_index[0]
            id1 = self.graph.edge_index[1]



            id0 = id0[merge]
            id1 = id1[merge]
            dH = dH[merge]


            _, dH_amax = torch_scatter.scatter_max(dH, id0)
            #dh_amax[i] = (argmax_j[dH[j]] for id0[j] = i) if \exist i in id0, else dH.shape + 1
            #dH_amax[i] is unique



            dH_amax = dH_amax[dH_amax < dH.shape[0]] # then dH_amax is a unique set

            operated_cnt = int(dH_amax.shape[0]) # cuda synchronized
            # print(operated_cnt)
            if operated_cnt == 0:
                continue



            max_operate_cnt = math.ceil((self.graph.num_nodes - min_com) * p)

            # print("max_op:"+str(max_operate_cnt))
            # print(max_operate_cnt)
            # print(operated_cnt)

            if operated_cnt > max_operate_cnt and di_max:
                id0 = id0[dH_amax]
                id1 = id1[dH_amax]
                dH = dH[dH_amax]
                _, dH_amax = torch_scatter.scatter_max(dH, id1)
                dH_amax = dH_amax[dH_amax < dH.shape[0]]
                operated_cnt = int(dH_amax.shape[0])


            if operated_cnt > max_operate_cnt:
                _, idx = torch.sort(dH[dH_amax], descending=True)
                # idx = torch.randperm(dH_amax.shape[0])
                dH_amax = dH_amax[idx[:max_operate_cnt]]
                operated_cnt = max_operate_cnt
                # p = 1.0

            # operated_cnt = ddd_new

            trans = torch.arange(self.graph.num_nodes, device=self.device)

            ids = id0[dH_amax]
            idt = id1[dH_amax]


            trans[ids] = trans[idt]

            if operated_cnt < 10:
                altered = torch.zeros(self.graph.num_nodes, device=self.device, dtype=torch.bool)
                altered[ids] = True
                altered[idt] = True
                altered_e = torch.logical_or(altered[self.graph.edge_index[0]], altered[self.graph.edge_index[1]])  # edges
            else:
                altered_e = None
            # trans[id0] = id1
            # trans[i] = j: label node i to j
            # print(operated_cnt)
            lg_merge = math.log2(operated_cnt + 2)

            #todo: test speed: limit var
            # var = #trans != torch.arange(self.graph.num_nodes, device=self.device)
            for i in range(int(lg_merge)):
                # ids[var] = trans[trans[var]]
                trans[ids] = trans[trans[ids]]


            # print(time.time() - t)

            # torch.tensor([id0])
            # torch.tensor([id1])


            # t = time.time()
            # failed = torch.logical_or(trans == torch.arange(self.graph.num_nodes) , altered) == 0
            # print(ids)
            # print(idt)
            # print(torch.nonzero(failed))


            # print(torch.nonzero(altered))
            # print(torch.nonzero(trans != torch.arange(self.graph.num_nodes)))

            trans = torch.unique(trans, return_inverse = True)[1]

            self.community_result = trans
            transs.append(trans)

            if self.graph.num_nodes - operated_cnt == min_com:
                break

            if adj_cover is None:
                cache = False
                self.graph = self.generate_hyper_graph(reset_community_id=False)


            else:
                cache = False
                self.graph, adj_cover= self.generate_hyper_graph(reset_community_id=False, edges_x=adj_cover)
                # self.graph.edge_weight = self.graph.edge_weight + f_hyper_edge(self.graph.x,
                #                                                                self.graph.edge_index[0],
                #                                                                self.graph.edge_index[1],
                #                                                                self.graph.edge_weight,
                #                                                                adj_cover)

            loop0 = True
            # break


            # print(time.time() - t)t

        if len(transs) != 0:
            trans = transs[-1]
            for i in reversed(range(len(transs) - 1)):
                trans = trans[transs[i]]

            self.community_result = trans
        else:
            self.community_result = com0

        self.graph = graph_old
        self.graph.x = graph_old_x

        # print(self.calc_entropy(log_M=self._log2m))

        # return


    def calc_entropy_rate(self, reduction = True):
        v1 = cluster_sum(self.graph.edge_index[0], self.graph.edge_weight).reshape(-1)
        norm_s = self.graph.edge_weight / v1[self.graph.edge_index[0]]
        if reduction:
            return torch.sum(v1 * cluster_sum(self.graph.edge_index[0], -norm_s * torch.log2(torch.clip(norm_s, min = self._eps)))) / v1.sum()
        return v1 * cluster_sum(self.graph.edge_index[0], -norm_s * torch.log2(norm_s)) / v1.sum()





def test():
    import torch

    t = time.time()
    torch.random.manual_seed(123)

    # edges = torch.tensor([[0,1],[1,2],[2,3], [3,0], [1,0],[2,1],[3,2], [0,3]],dtype=torch.int64).T
    # print(edges)
    # weight = torch.tensor([4,2,4,2] *2, dtype=torch.float32)
    # print(weight)
    # A critical point for complexity: E / V ~ sqrt(E) ?
    # edges = torch.randint(0, 300*400, (2, 300*400*5))
    edges = torch.randint(0, 10, (2, 50))
    weight = torch.rand(50)
    edges[[1, 0], 1::2] = edges[:, ::2]
    weight[1::2] = weight[::2]

    #mat = np.load("adj_mat.npy")
    # mat = np.abs(np.random.randn(100, 100))
    # mat = mat + mat.transpose()
    N = 1000
    e1 = []
    e2 = []
    w = []
    for ii in range(N):
        for jj in range(N):
            if ii != jj:
                e1.append(ii)
                e2.append(jj)
                w += [1]
    # e1 = [0,1,2,3,4,5]
    # e2 = [1,2,0,4,5,3]
    # e1, e2 = e1 + e2, e2 + e1
    #
    # w = [1,1,1,1,1, 1]
    # w = w + w
    edges = [torch.LongTensor(e1), torch.LongTensor(e2)]
    w = torch.Tensor(w)

    g = Graph(edge_index=edges, edge_weight = w)
    # N = 10
    # es = torch.arange(0, N, dtype=torch.int64).repeat(N)
    # et = torch.arange(0, N, dtype=torch.int64).repeat_interleave(N)
    # g = Graph(edge_index=(es, et))
    spmm(g, torch.rand(g.num_nodes, 1))
    # g.edge_weight = w
    # g.edge_weight = torch.rand(g.edge_weight.shape)

    print("Build Graph & init spmm: {}s".format(str(time.time() - t)))
    #g.remove_self_loops()
    t = time.time()

    gnx = g.to_networkx()

    print("Build NetworkX: {}s".format(str(time.time() - t)))
    t = time.time()

    proc = TwoDimSE(g)
    proc.force_coding(torch.arange(g.num_nodes))
    print("torch. H2 = " + str(proc.calc_entropy()))
    print("torch. H1 = " + str(proc.calc_e1(loop_community=True)))
    print("Initialize TwoDimSE: {}s".format(str(time.time() - t)))
    t = time.time()
    proc.process()
    print("Generate Encoding Tree: {}s".format(str(time.time() - t)))
    print("H1 = " + str(proc.entropy1))
    print("H2 = " + str(proc.entropy2))
    # print(proc._calc_community_tensor())
    t = time.time()

    print("torch. H2 = " + str(proc.calc_entropy()))

    print("torch. H(X1|X) = " + str(proc.calc_entropy_rate()))
    print("Initialize edge relations & Torch.forward: {}s".format(str(time.time() - t)))
    t = time.time()

    print("torch. H2 = " + str(proc.calc_entropy()))
    print("Torch.forward twice: {}s".format(str(time.time() - t)))
    t = time.time()

    import networkx as nx
    import matplotlib.pyplot as plt
    community = proc.community_result.cpu().numpy()

    labels = {i: community[i] for i in range(proc.node_cnt)}

    pos = nx.circular_layout(gnx)

    nx.draw_networkx_edges(gnx, pos, width=[float(d['weight'] * 1) for (u, v, d) in gnx.edges(data=True)])
    nx.draw_networkx_nodes(gnx, pos, node_color=community)
    nx.draw_networkx_labels(gnx, pos, labels=labels)
    # plt.title("H1 = " + str(proc.entropy1) + ",H2 = " + str(float(proc.calc_entropy())))
    plt.show()

    graph2 = proc.generate_hyper_graph()


    community2 = np.clip(community, a_min = 1, a_max=10000)
    proc.force_coding(torch.LongTensor(torch.tensor(community2)))
    proc2 = TwoDimSE(graph2)
    comh2 = torch.arange(graph2.num_nodes)
    comh2 = torch.clip(comh2, min = 1)
    proc2.force_coding(comh2)
    print(proc2.calc_eh())
    print( proc.calc_eh())

    nx.draw(graph2.to_networkx(), with_labels=True)
