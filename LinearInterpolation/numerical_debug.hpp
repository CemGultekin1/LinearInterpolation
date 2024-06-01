//
//  numerical_debug.hpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 4/13/24.
//

#ifndef numerical_debug_hpp
#define numerical_debug_hpp

#include <stdio.h>
#include "simplex_tree.hpp"

struct InterpolationTest{
    SimplexTree const* simplex_tree;
    InterpolationTest(SimplexTree const* simplex_tree):simplex_tree(simplex_tree){};
    float operator()(std::vector<float> & x);
};

#endif /* numerical_debug_hpp */
