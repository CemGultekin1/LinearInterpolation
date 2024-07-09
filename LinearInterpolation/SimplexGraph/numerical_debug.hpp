//
//  numerical_debug.hpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 4/13/24.
//

#ifndef numerical_debug_hpp
#define numerical_debug_hpp

#include <stdio.h>
#include "simplex_graph.hpp"

struct InterpolationTest{
    SimplexGraph const* simplex_graph;
    InterpolationTest(SimplexGraph const* simplex_graph):simplex_graph(simplex_graph){};
    float operator()(std::vector<float> & x);
};

#endif /* numerical_debug_hpp */
