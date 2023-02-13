//
// Created by hujin on 2022/11/1.
//

#include "si.h"
#include "si_SL.h"
#ifdef WIN_BUILD
#define COMP_WIN _declspec(dllexport)
#else
#define COMP_WIN
#endif


extern "C" {
	COMP_WIN void si(UINT n, UINT cnt_e, const UINT* es, const UINT* et, const double* w, UINT* result) {
		auto c = Cluster(n);
		c.init(cnt_e, es, et, w);
		c.process(result);
	};

    COMP_WIN void si_directed(UINT n, UINT cnt_e, const UINT* es, const UINT* et, const double* p,  const double* stat_dist, UINT* result) {
        auto c = Cluster(n);
        c.init_directed(cnt_e, es, et, p, stat_dist);
        c.process(result);
    };

	COMP_WIN double si_tgt(UINT n, UINT cnt_e, const UINT* es, const UINT* et, const double* w, UINT* result, double global_var) {
		//        int com = -1, add = -5;
		//        do{
		//            Cluster c = Cluster(n);
		//            c.init(cnt_e, es, et, w);
		//            c.log2m += global_var;
		//            c.target_cluster = tgt_com;
		//            c.process(result);
		//            com = (int)c.cluster_count;
		//            add ++;
		//        } while (com > tgt_com);

		auto c = Cluster(n);
		c.init(cnt_e, es, et, w);
		c.log2m += global_var;
		c.process(result);
		return 0;
	};
	COMP_WIN double si_tgt_adj(UINT n, UINT cnt_e, const UINT* es, const UINT* et, const double* w, UINT* result, double global_var, const UINT* adj) {
		//        int com = -1, add = -5;
		//        do{
		//            Cluster c = Cluster(n);
		//            c.init(cnt_e, es, et, w);
		//            c.log2m += global_var;
		//            c.target_cluster = tgt_com;
		//            c.process(result);
		//            com = (int)c.cluster_count;
		//            add ++;
		//        } while (com > tgt_com);

		auto c = Cluster(n);
		c.init(cnt_e, es, et, w, adj);
		c.log2m += global_var;
		c.process(result);
		return 0;
	};

	int test() {
		return 123;
	}

	COMP_WIN void si_hyper(UINT n, UINT cnt_e, const UINT* es, const UINT* et, const double* w, UINT* result) {
		auto c = ClusterSelfLoop(n);
		c.process(cnt_e, es, et, w, result);
	};
	COMP_WIN void si_hyper_adj(UINT n, UINT cnt_e, const UINT* es, const UINT* et, const double* w, const UINT* adj, UINT* result) {
		auto c = ClusterSelfLoop(n);
		c.process(cnt_e, es, et, w, result, adj);
	};
};
