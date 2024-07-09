//
//  numerical_debug.cpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 4/13/24.
//

#include "numerical_debug.hpp"


float InterpolationTest::operator()(std::vector<float> & x){
    SparsePoint sp{simplex_tree->midpoint_table.max_dim};
    simplex_tree->operator()(x, sp);
    std::vector<float> z(x.size(),0.);
    for(size_t i = 0; i < sp.weights.size(); ++i){
        simplex_tree->midpoint_table.increment_by_weights(z,sp.nodes[i],sp.weights[i],1,1);
    }
    float error = 0.;
    for(size_t i = 0 ; i < x.size() ; ++i){
        error += std::pow(x[i] - z[i],2);
    }
    return error;
}
