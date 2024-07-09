//
//  debug_display.hpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 3/31/24.
//

#ifndef debug_display_h
#define debug_display_h

#include <stdio.h>
#include <vector>
#include <iostream>
#include "graph_descend.hpp"
#include <iomanip>

template<typename T>
void print_vector(const std::vector<T>& vec,const std::string& title){
    std::cout << title << std::endl;
    for (T element : vec) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
}

void print_midpoint(SparsePoint& sp,const std::string& prior,const std::string& posterior){
    auto weights = &sp.weights;
    auto nodes = &sp.nodes;
    std::cout << prior << std::endl;
    for(int i = 0 ; i < nodes->size(); ++i){
        std::cout << nodes->at(i) << " : " << weights->at(i) << std::endl;
    }
    std::cout <<posterior <<  std::endl;
}
template<typename T>
void sciprint(T num, int precision = 2){
    std::cout << std::setprecision(precision) << std::scientific << num;
}

#endif /* debug_display_hpp */
