//
// Created by 17526 on 2022/4/17.
//

#ifndef SI_SI_SL_H
#define SI_SI_SL_H
#define UINT unsigned int

class ClusterSelfLoop {
	std::vector<double> community_vol{}, community_g{}, community_g0{};
	std::vector<UINT> parent{};
	std::vector<std::set<UINT> > community_adj_raw;
	std::vector<std::set<UINT>*> community_adj;
	std::vector<std::set<UINT> > merge_restriction_raw;
	std::vector<std::set<UINT>*> merge_restriction;
	bool use_restriction = false;

	std::unordered_map<UINT, std::pair<UINT, UINT> > merge_history;
	double log2m = 0, sum_degress2 = 0;

	UINT cluster_count{}, target_cluster{}, node_cnt{}, community_pos{};

	std::priority_queue<std::pair<double, std::pair<UINT, UINT> > > process_queue;

public:
	// memory bottleneck
	std::vector<std::unordered_map<UINT, double>> community_edges;
	double entropy2 = 0; //△H2
	explicit ClusterSelfLoop(UINT nodeCnt) : node_cnt(nodeCnt) {
		community_vol.resize(2 * nodeCnt);
		community_g.resize(2 * nodeCnt);
		community_g0.resize(2 * nodeCnt);
		parent.resize(2 * nodeCnt);
		community_adj_raw.resize(2 * nodeCnt);
		merge_restriction_raw.resize(2 * nodeCnt);
		community_adj.resize(2 * nodeCnt);
		merge_restriction_raw.resize(2 * nodeCnt);
		community_edges.resize(2 * nodeCnt);
	}

	virtual ~ClusterSelfLoop();

public:
	void process(UINT edgeCnt, const UINT* es, const UINT* et, const double* w, UINT* result, const UINT* adj = nullptr);

private:
	UINT getParent(UINT f);

	double get_edge_xy_recur(unsigned int i, unsigned int j, bool lastI, bool top);

	double get_edge_xy(unsigned int i, unsigned int j);

	double get_dH_residual(unsigned int i, unsigned int j, double edge, double& vj, double& gj, double& g0j, double& h2j, bool& calced);

	double get_dH(unsigned int i, unsigned int j);

	void get_vg(unsigned int i, unsigned int j, double& gx, double& vx, double& g0);

	void merge(unsigned int i, unsigned int j, double dH);

	void merge_try(unsigned int i, unsigned int j, double dH);

	void process_submodule();
};



#endif //SI_SI_H
