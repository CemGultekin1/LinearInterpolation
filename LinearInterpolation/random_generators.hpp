//
//  random_generators.hpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 3/31/24.
//

#ifndef random_generators_hpp
#define random_generators_hpp

#include <stdio.h>

#endif /* random_generators_hpp */


#include <random>
#include <set>

float random_float(float min, float max);
long random_long(long min, long max);
void fill_vector_with_sum_to_one(std::vector<float>& vec, long size);
void fill_vector_with_distinct_random_long(std::vector<long>& vec, int size, long min_value, long max_value);
void fill_vector_with_random_selections(std::vector<long>& fill_into_vec,std::vector<long>& select_from_vec, int size);
void random_float_vector(std::vector<float> * x,size_t n);


