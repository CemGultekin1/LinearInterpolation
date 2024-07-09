//
//  path_hash.cpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 4/3/24.
//

#include "graph_path.hpp"


void hash_forward(uint64_t& hash,long x){
//    hash = std::rotl(hash,4);
    hash ^= x;
}

void hash_backward(uint64_t& hash,long x){
    hash ^= x;
//    hash = std::rotr(hash,4);
}
void MidpointStep::set_escape_node(long new_escape_node){
    if(new_escape_node == escape_node){
        return;
    }
    escape_index = std::numeric_limits<size_t>::max();
    escape_alias = std::numeric_limits<long>::max();
    escape_node = new_escape_node;
}
MidpointStep::MidpointStep(){
    midpoint_alias = std::numeric_limits<size_t>::max();
    midpoint_node = std::numeric_limits<long>::max();
    
    escape_index = std::numeric_limits<size_t>::max();
    
    escape_node = std::numeric_limits<long>::max();
    escape_alias = std::numeric_limits<size_t>::max();    
}

std::string MidpointStep::to_string() const{
    return "(" + std::to_string(midpoint_alias)+"|" + std::to_string(escape_node) + "->" + std::to_string(midpoint_node) + ")";
}

HashableGraphPath::HashableGraphPath():std::vector<MidpointStep>(){
    hash_value = 0;
    
}

HashableGraphPath::HashableGraphPath(const HashableGraphPath& hash_graph_path):std::vector<MidpointStep>(hash_graph_path.size()){
    hash_value = hash_graph_path.hash_value;
    for(int i = 0; i < hash_graph_path.size(); ++i){
        this->at(i) = hash_graph_path.at(i);
    }
}
void HashableGraphPath::push_back(MidpointStep midpoint_step){
    std::vector<MidpointStep>::push_back(midpoint_step);
    auto escape = midpoint_step.escape_node;
    auto midpoint = midpoint_step.midpoint_node;
    hash_forward(hash_value, escape);
    hash_forward(hash_value, midpoint);
}
void HashableGraphPath::pop_back(){
    auto midstp = HashableGraphPath::back();
    auto midpoint = midstp.midpoint_node;
    auto escape = midstp.escape_node;
    hash_backward(hash_value, midpoint);
    hash_backward(hash_value, escape);
    std::vector<MidpointStep>::pop_back();
}

uint64_t HashableGraphPath::hash() const{
    return hash_value;
}


std::string HashableGraphPath::to_string() const{
    std::string x = "";
    for(const auto& p: *this){
        x += p.to_string()  + ",";
    }
    return x;
}
