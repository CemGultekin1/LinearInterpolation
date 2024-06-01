//
//  simplex_tree.cpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 3/31/24.
//

#include "graph_descend.hpp"
#include <assert.h>
#include "chrono_methods.hpp"

PointWithDictionary::PointWithDictionary(const std::vector<float>& input){
    weights.reserve(input.size() + 1);
    std::copy(std::begin(input),std::end(input),std::back_inserter(weights));
}

PointWithDictionary::PointWithDictionary(const size_t& max_dim){
    weights.reserve(max_dim);
}

void PointWithDictionary::to_point(SparsePoint & empty_point,
                                   float sparsification_tolerance,
                                   int num_threads,
                                   bool apply_sparsification) const{
    if(apply_sparsification){
        parallel_sparsification(weights,node2index,empty_point.weights, empty_point.nodes, sparsification_tolerance,num_threads);
    }else{
        for(const auto& [node,i]: node2index){
            empty_point.weights.push_back(weights.at(i));
            empty_point.nodes.push_back(node);
        }
    }
    
}

void PointWithDictionary::fill_up_default_coords(){
    node2index.insert({-1,weights.size() - 1});
    for(int i = 0; i < weights.size() -1; ++i){
        node2index.insert({-i-2,i});
    }
}

std::string PointWithDictionary::to_string() const{
    std::string x{};
    for(const auto & [node,index]: node2index){
        x += std::to_string(node) + ":" + std::to_string(weights.at(index)) + ",";
    }
    return x;
}
SparsePoint::SparsePoint(const size_t max_dim){
    weights.reserve(max_dim);
    nodes.reserve(max_dim);
}

SparsePoint::SparsePoint(const SparsePoint& sp){
    weights.reserve(sp.weights.size());
    nodes.reserve(sp.nodes.size());
    std::copy(sp.weights.begin(),sp.weights.end(),std::back_inserter(weights));
    std::copy(sp.nodes.begin(),sp.nodes.end(),std::back_inserter(nodes));
}

std::string SparsePoint::to_string() const{
    std::string x{};
    for(size_t i = 0; i < nodes.size(); i ++){
        x += std::to_string(nodes.at(i)) + ":" + std::to_string(weights.at(i)) + ",";
    }
    return x;
}

MidpointTable::MidpointTable(){
    alias2node =std::vector<long>{};
    node2aliases = std::unordered_map<long,std::vector<size_t>>{};
    data = std::vector<std::vector<float>>{};
    midpoints = std::vector<SparsePoint>{};
    hash2midpoint = std::unordered_map<uint64_t,std::vector<size_t>>{};
    facing_nodes = std::unordered_map<size_t, size_t>{};
    max_dim = 0;
    depth = 0;
}

long MidpointTable::add_node(const std::vector<float>& x){
    data.push_back(x);
    return data.size() - 1;
}


bool MidpointTable::check_step_consistency(const MidpointStep& ms) const{
    auto nodes = midpoints.at(ms.midpoint_alias).nodes;
    return std::find(nodes.begin(),nodes.end(),ms.escape_node) != nodes.end();
}

size_t MidpointTable::add_midpoint(const SparsePoint & p, size_t x_dim,long node){
    midpoints.push_back(p);
    auto alias = midpoints.size() - 1;
    max_dim = std::max(x_dim,max_dim);
    associate(node, alias);
    return alias;
}
void MidpointTable::update_depth(const HashableGraphPath& path){
    depth = std::max(path.size() + 1,depth);
}
size_t MidpointTable::find_midpoint_alias(const std::unordered_map<long,size_t>& node2index, const HashableGraphPath & descend) const{
    size_t min_alias = 0;
    if(!descend.empty()){
        min_alias = descend.back().midpoint_alias + 1;
    }
    for(size_t alias = min_alias ; alias < midpoints.size(); alias ++ ){
        if(parallel_set_contains(node2index, midpoints[alias].nodes)){
            return alias;
        }
    }
    return std::numeric_limits<size_t>::max();
}

void MidpointTable::insert_midpoint_on_hash(const uint64_t& hash_value, const size_t& alias){
    if(hash2midpoint.find(hash_value) == hash2midpoint.end()){
        hash2midpoint.insert({hash_value,std::vector<size_t>{}});
    }
    auto find = hash2midpoint.find(hash_value);
    
    if(!find->second.empty() && find->second.back() == alias){
        return;
    }
    find->second.push_back(alias);
}


size_t full_lookup_midpoint_alias_search(const std::vector<SparsePoint>& midpoints, const std::unordered_set<long>& nodes, const HashableGraphPath & descend){
    size_t min_alias = 0;
    if(!descend.empty()){
        min_alias = descend.back().midpoint_alias + 1;
    }
    for(size_t alias = min_alias ; alias < midpoints.size(); alias ++ ){
        if(parallel_set_contains(nodes, midpoints[alias].nodes)){
            return alias;
        }
    }
    return std::numeric_limits<size_t>::max();
}
size_t hashed_lookup_midpoint_alias_search(MidpointTable const* midpoint_table, const std::unordered_set<long>& nodes, const HashableGraphPath & descend){
    auto hv = descend.hash();
    auto pntr = midpoint_table->hash2midpoint.find(hv);
    if(pntr == midpoint_table->hash2midpoint.end()){
        return std::numeric_limits<size_t>::max();
    }
    size_t min_alias = 0;
    if(!descend.empty()){
        min_alias = descend.back().midpoint_alias + 1;
    }
/*
 A leaf simplex might have the same alias as a non-leaf simplex, hence it stillneeds to be
 checked for containement
 */

//    if(pntr->second.size() == 1){
//        auto alias =  pntr->second[0];
//        if(alias < min_alias){
//            return std::numeric_limits<size_t>::max();
//        }
//        return alias;
//    }
    
    for(const auto&  alias: pntr->second){
        if(alias < min_alias){
            continue;
        }
        if(parallel_set_contains(nodes, midpoint_table->midpoints[alias].nodes)){
            return alias;
        }
    }
    return std::numeric_limits<size_t>::max();
}
size_t MidpointTable::find_midpoint_alias(const std::unordered_set<long>& nodes, const HashableGraphPath & descend) const{
//    auto true_alias =  full_lookup_midpoint_alias_search(midpoints,nodes,descend);
    auto pred_alias = hashed_lookup_midpoint_alias_search(this,nodes,descend);
//    assert(true_alias == pred_alias);
//    return true_alias;
    return pred_alias;
}


void MidpointTable::associate(long node,size_t alias){
    alias2node.resize(std::max(alias2node.size(),alias+1),std::numeric_limits<long>::max());
    alias2node[alias] = node;
    if(node2aliases.find(node) == node2aliases.end()){
        node2aliases[node] = std::vector<size_t>{};
    }
    node2aliases[node].push_back(alias);
}


void MidpointTable::increment_by_weights(std::vector<float>& x, long node,float weight,int power,int num_threads) const{
    if(node > - 1){
        if(x.size() < data[node].size()){
            x.resize(data[node].size(),0.);
        }
        parallel_pointwise_addition(x,data[node],weight,power,num_threads);
//        if(power != 1){
//            for(size_t i = 0; i < data[node].size();++i){
//                x[i] += weight*std::pow(data[node][i],power);
//            }
//        }else{
//            for(size_t i = 0; i < data[node].size();++i){
//                x[i] += weight*data[node][i];
//            }
//        }
        
    }else{
        if(x.size() < max_dim - 1){
            x.resize(max_dim - 1,0.);
        }
        auto c = -node - 2;
        if(c < 0){
            return;
        }
        x[c] += weight*x.size();
    }
    
}


TreeDescend::TreeDescend(PointWithDictionary& x, const MidpointTable& _midpoint_table){
    data = &x;
    midpoint_table = &_midpoint_table;
}


void TreeDescend::descend(int num_threads){
    parallel_apply_first_layer(data->weights, num_threads);
    data->fill_up_default_coords();
    auto midpoint_alias = midpoint_table->find_midpoint_alias(data->node2index, descend_path);
    std::unordered_map<long,size_t> aliases{};
    while(midpoint_alias != std::numeric_limits<size_t>::max()){
        SparsePoint const* midpoint = &midpoint_table->midpoints[midpoint_alias];
        size_t escape_index = parallel_find_escape_node(
                            midpoint->weights,
                            midpoint->nodes,
                            data->weights,
                            data->node2index,
                                  num_threads);
        long midpoint_node = midpoint_table->alias2node[midpoint_alias];
        parallel_apply_midpoint_operation(
                                          midpoint->weights,
                                          midpoint->nodes,
                                          data->weights,
                                          data->node2index,
                                          escape_index,
                                          midpoint_node,
                                          num_threads
                                          );
        auto escape_node = midpoint->nodes.at(escape_index);
        size_t escape_alias = escape_node;
        if(aliases.find(escape_node) == aliases.end()){
            escape_alias = aliases[escape_node];
        }
        MidpointStep mstp(midpoint_node,midpoint_alias,escape_index,escape_node,escape_alias);
        descend_path.push_back(mstp);
        midpoint_alias = midpoint_table->find_midpoint_alias(data->node2index, descend_path);
    }
}


void TreeDescend::follow_midpoint_with_step_discovery(PointWithDictionary* _data,int num_threads){
    parallel_apply_first_layer(_data->weights, num_threads);
    _data->fill_up_default_coords();
    for(int i = 0; i< descend_path.size(); ++i){
        
        
        auto midpoint_step = descend_path.at(i);
        
        
        auto midpoint = midpoint_table->midpoints[midpoint_step.midpoint_alias];
        size_t escape_index = midpoint_step.escape_index;
        if(escape_index == std::numeric_limits<size_t>::max()){
            log_sink.start_event("follow_midpoint_with_step_discovery/find");
            auto it = std::find(midpoint.nodes.begin(),midpoint.nodes.end(),midpoint_step.escape_node);
            assert(it != midpoint.nodes.end());
            escape_index = it - midpoint.nodes.begin();
            descend_path.at(i).escape_index = escape_index;
            if(_data->node2index.find(midpoint_step.escape_node) == _data->node2index.end()){
                throw std::invalid_argument("");
            }
            log_sink.finish_event("follow_midpoint_with_step_discovery/find");
        }
        
        log_sink.start_event("follow_midpoint_with_step_discovery/parallel_apply_midpoint_operation");
        parallel_apply_midpoint_operation(
                                          midpoint.weights,
                                          midpoint.nodes,
                                          _data->weights,
                                          _data->node2index,
                                          escape_index,
                                          midpoint_step.midpoint_node,
                                          num_threads
                                          );
        log_sink.finish_event("follow_midpoint_with_step_discovery/parallel_apply_midpoint_operation");
    }
    
}

float TreeDescend::determinant() const{
    float det = 1.;
    for(const auto & midpoint_step: descend_path){
        auto midpoint = &midpoint_table->midpoints[midpoint_step.midpoint_alias];
        det *= midpoint->weights.at(midpoint_step.escape_index);
    }
    return det;
}
