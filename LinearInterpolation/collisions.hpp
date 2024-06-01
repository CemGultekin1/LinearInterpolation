//
//  collisions.hpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 4/3/24.
//

#ifndef collisions_hpp
#define collisions_hpp

#include <stdio.h>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <deque>
#include "graph_descend.hpp"
#include <tuple>



struct Simplex{
    std::unordered_set<long> nodes;
    HashableGraphPath descend_path;
    Simplex(HashableGraphPath const*, const std::unordered_map<long,size_t>&);
    Simplex(Simplex const*);
    Simplex();
    Simplex(size_t max_dim);
    Simplex(HashableGraphPath const*,size_t max_dim);
    Simplex(HashableGraphPath const*, const std::vector<long>&);
    bool not_root() const;
    MidpointStep back_track();
    void forward(MidpointStep);
    std::string to_string();
};


struct MidpointStepMultiplicity{
    size_t midpoint_alias;
    long midpoint_node;
    std::vector<long> escape_nodes;
    MidpointStepMultiplicity(long midpoint_node,size_t midpoint_alias);
};

void simplex_tree_ascend(MidpointTable& midpoint_table,
                         Simplex& leaf_simplex,
                         Simplex& collision_root_simplex,
                         std::deque<MidpointStepMultiplicity>& multiplicities);




struct SimplexIterator{
    Simplex* leaf_simplex;
    MidpointTable* midpoint_table;
    Simplex* collision_root;
    std::deque<MidpointStepMultiplicity> base_multiplicities;
    std::unordered_map<size_t,std::pair<MidpointStepMultiplicity,bool>> discovered_midpoint_steps;
    size_t max_alias;
    std::vector<std::pair<size_t,size_t>> multiplicity_counter_stack;
    SimplexIterator(Simplex* leaf_simplex,
                    MidpointTable* midpoint_table,
                    Simplex* collision_root,
                    std::deque<MidpointStepMultiplicity>* multiplicities,
                    size_t max_alias);
    bool foward();
    bool depleted();
};

#endif /* collisions_hpp */
