//
// Created by 17526 on 2022/4/17.
//
#include <vector>
#include <set>
#include <unordered_map>
#include <queue>


#ifndef SI_SI_H
#define SI_SI_H
#define UINT unsigned int

class Cluster {

protected:
	std::vector<double> community_vol{}, community_g{};
	std::vector<UINT> parent{};
	std::vector<std::set<UINT> > community_adj_raw;
	std::vector<std::set<UINT>*> community_adj;

	bool cover_adj = false;

	std::unordered_map<UINT, std::pair<UINT, UINT> > merge_history;

	UINT  node_cnt{}, community_pos{};

	std::priority_queue<std::pair<double, std::pair<UINT, UINT> > > process_queue;

public:
	// memory bottleneck
	std::vector<std::unordered_map<UINT, double>> community_edges;
	double entropy2 = 0; //△H2
	double log2m = 0, sum_degress2 = 0;
	UINT target_cluster{}, cluster_count{};

	explicit Cluster(UINT nodeCnt) : node_cnt(nodeCnt) {
		community_vol.resize(2 * nodeCnt);
		community_g.resize(2 * nodeCnt);
		parent.resize(2 * nodeCnt);
		community_adj_raw.resize(2 * nodeCnt);
		community_adj.resize(2 * nodeCnt);
		community_edges.resize(2 * nodeCnt);
	}

	virtual ~Cluster();

    double calc_si1();

public:
	void process(UINT* result);

	void init(unsigned int cnt_e, const unsigned int* es, const unsigned int* et, const double* w, const UINT* adj = nullptr);

    void init_directed(unsigned int cnt_e, const unsigned int *es, const unsigned int *et, const double *p_ij,
                       const double *stat_dist, const UINT* adj = nullptr);

	double calc_si(unsigned int cnt_e, const unsigned int* es, const unsigned int* et, const double* w);

protected:
	UINT getParent(UINT f);

	double get_edge_xy_recur(unsigned int i, unsigned int j, bool lastI, bool top);

	double get_edge_xy(unsigned int i, unsigned int j);

	inline double get_dH_residual(unsigned int i, unsigned int j, double edge, double& vj, double& gj, double& h2j, bool& calced);

	inline double get_dH(unsigned int i, unsigned int j);

	void get_vg(unsigned int i, unsigned int j, double& gx, double& vx);

    virtual void merge(unsigned int i, unsigned int j, double dH);

	void process_submodule();

};



class Cluster_3Merge: public Cluster{
public:
    explicit Cluster_3Merge(UINT nodeCnt) :Cluster(nodeCnt){};
protected:
    void merge(unsigned int i, unsigned int j, double dH) override;
};


#endif //SI_SI_H
