//
//  random_generators.cpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 3/31/24.
//

#include "random_generators.hpp"


float random_float(float min, float max) {
    return ((float) rand())*(max - min)/((float)(RAND_MAX));
}

void random_float_vector(std::vector<float> * x,size_t n) {
    for(int i = 0; i < n ; i++){
        x->push_back(random_float(0,1));
    }
}



long random_long(long min, long max) {
    return ((long) rand())*(max - min)/((long)(RAND_MAX));
}



void fill_vector_with_sum_to_one(std::vector<float>& vec, long size) {
    vec.clear();
    vec.reserve(size);

    float sum = 0.0;
    for (int i = 0; i < size; ++i) {
        float rand_value = random_float(0.0, 1.0);
        vec.push_back(rand_value);
        sum += rand_value;
    }
    for(int i = 0; i < size ; ++i){
        vec[i] = vec[i]/sum;
    }
}


void fill_vector_with_distinct_random_long(std::vector<long>& vec, int size, long min_value, long max_value) {
    vec.clear();
    vec.reserve(size);

    std::set<long> generated_numbers;
    while (vec.size() < size) {
        long rand_value = random_long(min_value, max_value);
        if (generated_numbers.insert(rand_value).second) {
            vec.push_back(rand_value);
        }
    }
}


void fill_vector_with_random_selections(std::vector<long>& fill_into_vec,std::vector<long>& select_from_vec, int size){
    fill_into_vec.clear();
    fill_into_vec.reserve(size);
    
    std::vector<long> inds{};
    inds.reserve(size);
    fill_vector_with_distinct_random_long(inds,size,0,select_from_vec.size()-1);
    for(long i = 0; i < size; i ++ ){
        fill_into_vec.push_back(select_from_vec[inds[i]]);
    }
}
