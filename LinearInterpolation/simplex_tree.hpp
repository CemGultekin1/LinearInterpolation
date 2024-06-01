//
//  simplex_tree.hpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 4/2/24.
//

#ifndef simplex_tree_hpp
#define simplex_tree_hpp

#include <stdio.h>
#include "graph_descend.hpp"


struct SimplexTree{
    MidpointTable midpoint_table;
    int max_nthreads;
    float sparsification_tolerance;
    SimplexTree(int nthreads,float sparsification_tolerance);
    void operator()(std::vector<float>& x,SparsePoint& sp,bool sparse = true) const;
    size_t add_midpoint(const std::vector<float>& x);
};

#endif /* simplex_tree_hpp */
