//
//  point.cpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 6/5/24.
//

#include "graph_nodes.hpp"


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



//struct Point{
//    Point(
//};
