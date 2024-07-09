//
//  simplex_graph.hpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 4/2/24.
//

#ifndef simplex_graph_hpp
#define simplex_graph_hpp

#include <stdio.h>
#include "graph_descend.hpp"


struct SimplexGraph{
    MidpointTable midpoint_table;
    int max_nthreads;
    float sparsification_tolerance;
    SimplexGraph(int nthreads,float sparsification_tolerance);
    void operator()(std::vector<float>& x,SparsePoint& sp,bool sparse = true) const;
    size_t add_midpoint(const std::vector<float>& x);
};

#endif /* simplex_graph_hpp */
