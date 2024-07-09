//
//  linearAlgebra.hpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 3/31/24.
//

#ifndef numerical_ops_hpp
#define numerical_ops_hpp

#include <stdio.h>
#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include <limits>
#include <cmath>
#include <numeric>
#include <ranges>
#include "graph_path.hpp"
#include <unordered_set>

size_t parallel_find_escape_node(const std::vector<float>& midpoint_weights,
                           const std::vector<long>& midpoint_nodes,
                           const std::vector<float>& x_weights,
                           const std::unordered_map<long,unsigned long>& x_nodes_map,
                           int num_threads) ;
void parallel_apply_midpoint_operation(
                    const std::vector<float>& midpoint_weights,
                    const std::vector<long> &midpoint_nodes,
                    std::vector<float>& x_weights,
                    std::unordered_map<long,size_t>& x_nodes_map,
                    size_t escape_index,
                    long midpoint_node,
                    int num_threads);
void parallel_apply_first_layer(
                    std::vector<float>& x_weights,int num_threads);
void parallel_sparsification(const std::vector<float>& x_weights,
                         const std::unordered_map<long,size_t>& x_nodes,
                         std::vector<float>& new_x_weights,
                         std::vector<long>& new_x_nodes,
                         float tolerance,
                        int num_threads);
bool parallel_set_contains(const std::unordered_map<long,size_t>& x_nodes, const std::vector<long>& midpoint_nodes);
bool parallel_set_contains(const std::unordered_set<long>& x_nodes, const std::vector<long>& midpoint_nodes);
bool parallel_vector_contains(const long& node, const std::vector<long>& nodes);
void parallel_pointwise_addition(std::vector<float>& out,const std::vector<float>& inc,float weight,int power,int num_threads);
#endif /* linearAlgebra_hpp */
