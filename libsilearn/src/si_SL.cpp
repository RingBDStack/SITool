//
// Created by hujin on 2022/3/25.
//


#include <cmath>
#include <queue>
#include <vector>
#include <unordered_map>
#include <set>
#include <cassert>
#include <map>
#include <ctime>
#include "iostream"
#include "si_SL.h"

#define try_bound 0
#define eps 0
#define MERGE merge
//#define DELAY_CALC

#define double double
#define log2 log2
#define ULL unsigned long long
#define UINT unsigned int






UINT ClusterSelfLoop::getParent(UINT f) {
	if (parent[f] == f)
		return f;
	else return parent[f] = getParent(parent[f]);
}
double ClusterSelfLoop::get_edge_xy_recur(UINT i, UINT j, bool lastI, bool top = true) {
	if (community_edges[i].count(j))
		return community_edges[i][j];
	if (!lastI && merge_history.count(i)) {
		auto& p = merge_history[i];
		double result = get_edge_xy_recur(p.first, j, true) + get_edge_xy_recur(p.second, j, true, false);
		community_edges[i].erase(p.first);
		community_edges[i].erase(p.second);
		if (top)community_edges[i][j] = result;
		return result;
	}
	if (lastI && merge_history.count(j)) {
		auto& p = merge_history[j];
		double result = get_edge_xy_recur(i, p.first, true) + get_edge_xy_recur(i, p.second, false, false);
		community_edges[i].erase(p.first);
		community_edges[i].erase(p.second);
		if (top)community_edges[i][j] = result;
		return result;
	}
	return 0;
}
inline double ClusterSelfLoop::get_edge_xy(UINT i, UINT j) {
	if (community_edges[i].count(j))
		return community_edges[i][j];
#ifdef DELAY_CALC
	if (merge_history.count(i)) {
		return get_edge_xy_recur(i, j, false);
	}
	if (merge_history.count(j)) {
		return get_edge_xy_recur(i, j, true);
	}
#endif
	return 0;
}
inline double geth2(double v, double g, double g0) {
	return g0 == g ? 0 : (g0 - g) * log2(v);
}

double ClusterSelfLoop::get_dH_residual(UINT i, UINT j, double edge, double& vj, double& gj, double& g0j, double& h2j, bool& calced) {
	if (!calced) {
		vj = community_vol[j];
		gj = community_g[j];
		g0j = community_g0[j];
		h2j = geth2(vj, gj, g0j);
		calced = true;
	}
	double vi = community_vol[i];
	double gi = community_g[i];
	double g0i = community_g0[i];

	double vx = vi + vj;
	double gx = gi + gj - 2 * edge;
	double dH = (geth2(vi, gi, g0i) + h2j - geth2(vx, gx, g0i + g0j) +
		2 * edge * log2m);
	return dH;
}
double ClusterSelfLoop::get_dH(UINT i, UINT j) {
	double vi = community_vol[i], vj = community_vol[j];
	double gi = community_g[i], gj = community_g[j];
	double g0i = community_g[i], g0j = community_g0[j];

	double vx = vi + vj;
	double edge = get_edge_xy(i, j);
	double gx = gi + gj - 2 * edge;
	double g0x = g0i + g0j;
	double dH = (geth2(vi, gi, g0i) + geth2(vj, gj, g0j) - geth2(vx, gx, g0x) +
		2 * edge * log2m);
	return dH;
}
void ClusterSelfLoop::get_vg(UINT i, UINT j, double& gx, double& vx, double& g0x) {
	double vi = community_vol[i], vj = community_vol[j];
	double gi = community_g[i], gj = community_g[j];
	double g0i = community_g0[i], g0j = community_g0[j];
	double edge = get_edge_xy(i, j);

	vx = vi + vj;
	gx = gi + gj - 2 * edge;
	g0x = g0i + g0j;
}


void ClusterSelfLoop::merge(UINT i, UINT j, double dH) {

	UINT p = community_pos++;
	parent[i] = p;
	parent[j] = p;
	get_vg(i, j, community_g[p], community_vol[p], community_g0[p]);
	entropy2 -= dH;

#ifndef DELAY_CALC
	auto si = community_adj[i]->size();
	auto sj = community_adj[j]->size();
	if (si < sj) {
		community_adj[j]->insert(community_adj[i]->begin(), community_adj[i]->end());
		community_adj[i]->clear();
		community_adj[p] = community_adj[j];
		community_adj[p]->erase(j);
		community_adj[p]->erase(i);

	}
	else {
		community_adj[i]->insert(community_adj[j]->begin(), community_adj[j]->end());
		community_adj[j]->clear();
		community_adj[p] = community_adj[i];
		community_adj[p]->erase(j);
		community_adj[p]->erase(i);
	}

	bool calced = false;
	double vp, gp, g0p, h2p;

	for (auto x : *community_adj[p]) {
		double sum_e = get_edge_xy(i, x) + get_edge_xy(j, x);
		if (sum_e > 0) {
			community_edges[p][x] = sum_e;
			community_edges[x][p] = sum_e;
		}

		double dh = get_dH_residual(x, p, sum_e, vp, gp, g0p, h2p, calced);
		if (dh > eps || target_cluster > 0) {
			process_queue.push({ dh, {p, x} });
		}
		community_edges[x].erase(i);
		community_adj[x]->erase(i);
		community_edges[x].erase(j);
		community_adj[x]->erase(j);
		community_adj[x]->insert(p);
	}


	community_edges[i].clear();
	community_edges[j].clear();
#else
	merge_history[p] = { i, j };
#endif
}

inline void merge_subset(unsigned int i, unsigned int j, unsigned int new_pos, std::vector<std::set<UINT>*>& community_adj) {
	auto si = community_adj[i]->size();
	auto sj = community_adj[j]->size();
	if (si < sj) {
		community_adj[j]->insert(community_adj[i]->begin(), community_adj[i]->end());
		community_adj[i]->clear();
		community_adj[new_pos] = community_adj[j];
		community_adj[new_pos]->erase(j);
		community_adj[new_pos]->erase(i);

	}
	else {
		community_adj[i]->insert(community_adj[j]->begin(), community_adj[j]->end());
		community_adj[j]->clear();
		community_adj[new_pos] = community_adj[i];
		community_adj[new_pos]->erase(j);
		community_adj[new_pos]->erase(i);
	}
}

void ClusterSelfLoop::merge_try(UINT i, UINT j, double dH) {

	UINT p = community_pos++;
	parent[i] = p;
	parent[j] = p;
	get_vg(i, j, community_g[p], community_vol[p], community_g0[p]);
	entropy2 -= dH;

	bool revert = dH < eps;
	bool calced = false;
	double vp, gp, g0p, h2p;

	for (auto x : *community_adj[i]) {
		if (x == j) {
			continue;
		}
		double e = get_edge_xy(i, x) + get_edge_xy(j, x);
		double dh = get_dH_residual(x, p, e, vp, gp, g0p, h2p, calced);
		if (dh > try_bound * sum_degress2 && dh + dH > eps || target_cluster > 0) {
			process_queue.push({ dh, {p, x} });
			revert = false;
		}
	}
	for (auto x : *community_adj[j]) {
		if (x == i) {
			continue;
		}
		if (community_adj[i]->count(x)) {
			continue;
		}
		double e = get_edge_xy(i, x) + get_edge_xy(j, x);
		double dh = get_dH_residual(x, p, e, vp, gp, g0p, h2p, calced);
		if (dh > try_bound * sum_degress2 && dh + dH > eps || target_cluster > 0) {
			process_queue.push({ dh, {p, x} });
			revert = false;
		}
	}
	if (revert) {
		community_pos--;
		parent[i] = i;
		parent[j] = j;
		return;
	}

	merge_subset(i, j, p, community_adj);
	for (auto x : *community_adj[p]) {
		double sum_e = get_edge_xy(i, x) + get_edge_xy(j, x);
		if (sum_e > 0) {
			community_edges[p][x] = sum_e;
			community_edges[x][p] = sum_e;
		}
		community_adj[x]->erase(i);
		community_edges[x].erase(i);
		community_adj[x]->erase(j);
		community_edges[x].erase(j);
		community_adj[x]->insert(p);
	}

	community_edges[i].clear();
	community_edges[j].clear();
}

void ClusterSelfLoop::process_submodule() {
	while (!process_queue.empty() && target_cluster != cluster_count) {
		auto x = process_queue.top();
		process_queue.pop();
		unsigned int fx = getParent(x.second.first);
		unsigned int fy = getParent(x.second.second);
		if (fx == fy) {
			continue;
		}
		if (fx != x.second.first || fy != x.second.second) {
			double dH = get_dH(fx, fy);
			if (dH > 0)process_queue.push({ dH, {fx, fy} });
			continue;
		}
		cluster_count -= 1;
		MERGE(x.second.first, x.second.second, x.first);
		//        std::cout<<x.second.first<<" "<<x.second.second<<" "<<x.first<<std::endl;
	}
}

void ClusterSelfLoop::process(UINT cnt_e, const UINT* es, const UINT* et, const double* w, UINT* result, const UINT* adj) {
	for (UINT i = 0; i < cnt_e; ++i) {
		UINT s = es[i], t = et[i];
		community_vol[s] += w[i];
		if (s != t) {
			community_g[s] += w[i];
			community_g0[s] += w[i];
			community_edges[s][t] += w[i];
			if (adj == nullptr || adj[i]) {
				community_adj_raw[s].insert(t);
			}
		}
		sum_degress2 += w[i];
	}
	for (UINT i = 0; i < node_cnt; ++i) {
		parent[i] = i;
		community_adj[i] = &(community_adj_raw[i]);
	}
	for (UINT i = node_cnt; i < node_cnt * 2; ++i) {
		parent[i] = i;
		community_vol[i] = community_g[i] = community_g0[i] = 0;
	}
	community_pos = node_cnt;
	cluster_count = node_cnt;
	log2m = log2(sum_degress2);
	for (UINT i = 0; i < node_cnt; ++i) {
		for (auto j : community_adj_raw[i]) {
			double dh = get_dH(i, j);
			if ((dh > try_bound * sum_degress2 || target_cluster > 0) && i < j)
				process_queue.push({ dh, {i, j} });
		}
	}
#ifndef DELAY_CALC
	while (!process_queue.empty()) {
		auto x = process_queue.top();
		process_queue.pop();
		if (parent[x.second.first] != x.second.first || parent[x.second.second] != x.second.second) {
			continue;
		}
		if (target_cluster == cluster_count) {
			break;
		}
		cluster_count -= 1;
		MERGE(x.second.first, x.second.second, x.first);
		//        std::cout<<x.second.first<<" "<<x.second.second<<" "<<x.first<<std::endl;
	}
#else
	process_submodule();
#endif
	UINT pos_id = 0;
	std::map<UINT, UINT> used_id;
	for (UINT i = 0; i < node_cnt; ++i) {
		UINT id = getParent(i);
		if (used_id.count(id)) {
			result[i] = used_id[id];
		}
		else {
			result[i] = pos_id;
			used_id[id] = pos_id;
			pos_id++;
		}
	}

	std::priority_queue<std::pair<double, std::pair<UINT, UINT> > > process_queue_t;
	process_queue = process_queue_t;
	entropy2 /= sum_degress2;
}

ClusterSelfLoop::~ClusterSelfLoop() {
	for (auto& x : community_adj_raw) {
		std::set<UINT> k;
		x.swap(k);
	}
	for (auto& x : community_edges) {
		std::unordered_map<UINT, double> k;
		x.swap(k);
	}
}




//double gen_rand(){
//    double ret = 0;
//    for (UINT i = 0; i < 3; ++i) {
//        double x = (double )(rand() - rand())/ RAND_MAX;
//        ret -= x * x;
//    }
//    return exp2(ret);
//}
//int main(){
//    srand(123);
//    for (int t = 0; t < 1000; ++t) {
//        UINT imgsize = 256;
//        UINT n = imgsize * imgsize, ne = 8 * imgsize * imgsize, nn = imgsize;
//        UINT *es = static_cast<UINT *>(malloc(ne * sizeof(UINT)));
//        UINT *et = static_cast<UINT *>(malloc(ne * sizeof(UINT)));
//        auto *w = static_cast<double *>(malloc(ne * sizeof(double)));
//        auto *ans = static_cast<UINT *>(malloc(n * sizeof(UINT)));
//
//        for (UINT i = 0; i < n; ++i) {
//            es[i * 8] = i + 1 >= n ? i : i+1;
//            es[i * 8 + 1] = i + nn >= n ? i : i + nn;
//            es[i * 8 + 2] = i + nn + 1 >= n ? i : i + nn + 1;
//            es[i * 8 + 3] = i + nn - 1 >= n ? i : i + nn - 1;
//            es[i * 8 + 4] = i;
//            es[i * 8 + 5] = i;
//            es[i * 8 + 6] = i;
//            es[i * 8 + 7] = i;
//
//            et[i * 8] = i;
//            et[i * 8 + 1] = i;
//            et[i * 8 + 2] = i;
//            et[i * 8 + 3] = i ;
//            et[i * 8 + 4] = i + 1 >= n ? i : i+1;
//            et[i * 8 + 5] = i + nn >= n ? i : i + nn;
//            et[i * 8 + 6] = i + nn + 1 >= n ? i : i + nn + 1;
//            et[i * 8 + 7] = i + nn - 1 >= n ? i : i + nn - 1;
//
//
//            w[i * 8] = gen_rand();
//            w[i * 8+1] = gen_rand();
//            w[i * 8+2] = gen_rand();
//            w[i * 8+3] = gen_rand();
//            w[i * 8+4] = w[i * 8];
//            w[i * 8+5] = w[i * 8+1];
//            w[i * 8+6] = w[i * 8+2];
//            w[i * 8+7] = w[i * 8+3];
//        }
//        auto time = clock();
//
//        auto c = ClusterSelfLoop(n);
//        c.process(ne, es, et, w, ans);
//        std::cout<<(double )(clock() - time) /  CLOCKS_PER_SEC<< std::endl;
//        printf("%.15lf\n", c.entropy2);
////    UINT tot = 0;
////    for(UINT i = 0; i< n; i++){
////        tot += community_adj_raw[i].size();
////    }
////    printf("%d\n", tot);
//        printf("%lld\n", c.community_edges.size());
//    }
//    return 0;
//}
