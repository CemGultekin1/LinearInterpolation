//
//  resimplexification_scoring.hpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 4/10/24.
//

#ifndef resimplexification_hpp
#define resimplexification_hpp

#include <stdio.h>
#include "graph_descend.hpp"
#include "facing_node.hpp"

void l2_variation(const MidpointTable&  midpoint_table,FacingNodes& fn,int);


struct MomentCollector{
    MidpointTable const * midpoint_table;
    size_t counter;
    int num_threads;
    std::vector<std::vector<float>> moments;
    void remove(long node);
    void add(long node);
    float variance() const ;
    MomentCollector(size_t degree,const MidpointTable& midpoint_table,int nthreads);
};



void resimplexify(MidpointTable&  midpoint_table,const FacingNodes& fn,std::vector<FacingNodes>& fnheap,const facing_node_comparison& compr,int);

#endif /* resimplexification_scoring_hpp */
