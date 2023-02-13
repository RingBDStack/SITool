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
#include "si.h"

#define try_bound 1e-5
#define eps 0
#define MERGE merge
//#define DELAY_CALC

#define double double
#define log2 log2
#define ULL unsigned long long
#define UINT unsigned int






UINT Cluster::getParent(UINT f) {
	if (parent[f] == f)
		return f;
	else return parent[f] = getParent(parent[f]);
}
double Cluster::get_edge_xy_recur(UINT i, UINT j, bool lastI, bool top = true) {
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
inline double Cluster::get_edge_xy(UINT i, UINT j) {
	if (community_edges[i].count(j))
		return community_edges[i][j];
	//#ifdef DELAY_CALC
	if (merge_history.count(i)) {
		return get_edge_xy_recur(i, j, false);
	}
	if (merge_history.count(j)) {
		return get_edge_xy_recur(i, j, true);
	}
	//#endif
	return 0;
}

inline double geth2(double v, double g) {
	return v == g ? 0 : (v - g) * log2(v);
}

double Cluster::get_dH_residual(UINT i, UINT j, double edge, double& vj, double& gj, double& h2j, bool& calced) {
	if (!calced) {
		vj = community_vol[j];
		gj = community_g[j];
		h2j = geth2(vj, gj);
		calced = true;
	}
	double vi = community_vol[i];
	double gi = community_g[i];

	double vx = vi + vj;
	double gx = gi + gj - 2 * edge;
//	double dH = -1 / (geth2(vi, gi) + h2j - geth2(vx, gx)  ) * (
//		2 * edge * log2m);
    double dH =  (geth2(vi, gi) + h2j - geth2(vx, gx)  ) +
            2 * edge * log2m;
	return dH;
}
double Cluster::get_dH(UINT i, UINT j) {
	double vi = community_vol[i], vj = community_vol[j];
	double gi = community_g[i], gj = community_g[j];

	double vx = vi + vj;
	double edge = get_edge_xy(i, j);
	double gx = gi + gj - 2 * edge;
//	double dH = -1 / (geth2(vi, gi) + geth2(vj, gj) - geth2(vx, gx) ) * (
//		2 * edge * log2m);
    double dH = (geth2(vi, gi) + geth2(vj, gj) - geth2(vx, gx) ) +
            2 * edge * log2m;
	return dH;
}
void Cluster::get_vg(UINT i, UINT j, double& gx, double& vx) {
	double vi = community_vol[i], vj = community_vol[j];
	double gi = community_g[i], gj = community_g[j];
	double edge = get_edge_xy(i, j);

	vx = vi + vj;
	gx = gi + gj - 2 * edge;
}


void Cluster::merge(UINT i, UINT j, double dH) {

	UINT p = community_pos++;
	parent[i] = p;
	parent[j] = p;
	get_vg(i, j, community_g[p], community_vol[p]);
	entropy2 -= dH;

#ifndef DELAY_CALC
	auto si = community_adj[i]->size();
	auto sj = community_adj[j]->size();
	if (si < sj) {
		community_adj[j]->insert(community_adj[i]->begin(), community_adj[i]->end());
		community_adj[i]->clear();
		std::set<UINT>().swap(*(community_adj[i]));
		community_adj[p] = community_adj[j];
		community_adj[p]->erase(j);
		community_adj[p]->erase(i);
	}
	else {
		community_adj[i]->insert(community_adj[j]->begin(), community_adj[j]->end());
		community_adj[j]->clear();
		std::set<UINT>().swap(*(community_adj[j]));
		community_adj[p] = community_adj[i];
		community_adj[p]->erase(j);
		community_adj[p]->erase(i);
	}

	bool calced = false;
	double vp, gp, h2p;
	//	if (cover_adj) {
	//		for (int x = 0; x < p; x++) {
	//			if (getParent(x) != x) continue;
	//			double sum_e = get_edge_xy(i, x) + get_edge_xy(j, x);
	//			if (sum_e > 0) {
	//				community_edges[p][x] = sum_e;
	//				community_edges[x][p] = sum_e;
	//			}
	//			bool neighbor = community_adj[p]->count(x);
	//			double dh = get_dH_residual(x, p, sum_e, vp, gp, h2p, calced);
	//			if ((dh > eps || target_cluster > 0) && neighbor) {
	//				process_queue.push({ dh, {p, x} });
	//			}
	//			community_edges[x].erase(i);
	//			community_adj[x]->erase(i);
	//			community_edges[x].erase(j);
	//			community_adj[x]->erase(j);
	//			if (neighbor)community_adj[x]->insert(p);
	//		}
	//	}
	//	else
	for (auto x : *community_adj[p]) {
		double sum_e = get_edge_xy(i, x) + get_edge_xy(j, x);
		if (sum_e > 0) {
			community_edges[p][x] = sum_e;
			community_edges[x][p] = sum_e;
		}

		double dh = get_dH_residual(x, p, sum_e, vp, gp, h2p, calced);
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
	std::unordered_map<UINT, double>().swap(community_edges[i]);
	community_edges[j].clear();
	std::unordered_map<UINT, double>().swap(community_edges[j]);

	if (cover_adj)
		merge_history[p] = { i, j };
#else
	merge_history[p] = { i, j };
#endif
}

void Cluster_3Merge::merge(UINT i, UINT j, double dH) {

	UINT p = community_pos++;
	parent[i] = p;
	parent[j] = p;
	get_vg(i, j, community_g[p], community_vol[p]);
	entropy2 -= dH;

	bool revert = dH < eps;
	bool calced = false;
	double vp, gp, h2p;

	for (auto x : *community_adj[i]) {
		if (x == j) {
			continue;
		}
		double e = get_edge_xy(i, x) + get_edge_xy(j, x);
		double dh = get_dH_residual(x, p, e, vp, gp, h2p, calced);
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
		double dh = get_dH_residual(x, p, e, vp, gp, h2p, calced);
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


	if (cover_adj)
		merge_history[p] = { i, j };
}

void Cluster::process_submodule() {
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


void Cluster::init_directed(unsigned int cnt_e, const unsigned int *es, const unsigned int *et, const double *p, const double *stat_dist, const UINT* adj) {
    cover_adj = adj != nullptr;
    for (UINT i = 0; i < cnt_e; ++i) {
        UINT s = es[i], t = et[i];
        if (s != t) {
            community_g[t] += p[i]; //p(t)
            community_edges[s][t] += p[i]; // p(s, t)
            if (adj == nullptr || adj[i]) {
                community_adj_raw[s].insert(t);
            }
        }
    }
    for (UINT i = 0; i < node_cnt; ++i) {
        parent[i] = i;
        community_vol[i] += stat_dist[i] ; // v(s)
        sum_degress2 += stat_dist[i];
        community_adj[i] = &(community_adj_raw[i]);
    }
    for (UINT i = node_cnt; i < node_cnt * 2; ++i) {
        parent[i] = i;
        community_vol[i] = community_g[i] = 0;
    }
    community_pos = node_cnt;
    cluster_count = node_cnt;
    log2m = log2(sum_degress2);
}



void Cluster::init(UINT cnt_e, const UINT* es, const UINT* et, const double* w, const UINT* adj) {
	cover_adj = adj != nullptr;
	for (UINT i = 0; i < cnt_e; ++i) {
		UINT s = es[i], t = et[i];
		community_vol[s] += w[i] ; // v(s)
		if (s != t) {
			community_g[t] += w[i]; //p(t)
			community_edges[s][t] += w[i]; // p(s, t)
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
		community_vol[i] = community_g[i] = 0;
	}
	community_pos = node_cnt;
	cluster_count = node_cnt;
	log2m = log2(sum_degress2);
}
void Cluster::process(UINT* result) {
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
#ifndef WIN_BUILD
//		std::cout<<x.second.first<<" "<<x.second.second<<" "<<x.first<<std::endl;
#endif
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

double Cluster::calc_si(unsigned int cnt_e, const unsigned int* es, const unsigned int* et, const double* w) {
    double ret = 0, sw = 0;
    for(UINT i = 0; i < cnt_e; i++){
        UINT s = es[i], t = et[i];
        unsigned int ps = getParent(s);
        unsigned int pt = getParent(t);
        ret += w[i] * log2(community_vol[s] / community_vol[pt]);
        sw += w[i];
        if(ps != pt){
            ret += w[i] * log2(community_vol[pt] / sum_degress2);
        }
    }
    assert(std::abs(sw - sum_degress2) < 0.01);
    return -ret / sw;
}
double Cluster::calc_si1() {
    double ret = 0;
    for (int i = 0 ; i < node_cnt; i++){
        ret += log2(community_vol[i] / sum_degress2) * community_vol[i];
    }
    return -ret / sum_degress2;
}


Cluster::~Cluster() {
	for (auto& x : community_adj_raw) {
		std::set<UINT> k;
		x.swap(k);
	}
	for (auto& x : community_edges) {
		std::unordered_map<UINT, double> k;
		x.swap(k);
	}
}

