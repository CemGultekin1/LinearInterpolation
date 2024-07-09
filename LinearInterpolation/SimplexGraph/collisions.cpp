//
//  collisions.cpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 4/3/24.
//

#include "collisions.hpp"

void simplex_tree_ascend(MidpointTable& midpoint_table,
                         Simplex& leaf_simplex,
                         Simplex& collision_root_simplex,
                         std::deque<MidpointStepMultiplicity>& multiplicities){
    auto max_dim = midpoint_table.max_dim;
    while(!leaf_simplex.descend_path.empty()){
        MidpointStep mdstp = leaf_simplex.back_track();
        
        auto it = collision_root_simplex.nodes.find(mdstp.midpoint_node);
        if(it != collision_root_simplex.nodes.end()){
            auto nodes = &midpoint_table.midpoints[mdstp.midpoint_alias].nodes;
            MidpointStepMultiplicity mstp1{mdstp.midpoint_node,mdstp.midpoint_alias};
            for(auto n: *nodes){
                auto it1 =collision_root_simplex.nodes.find(n);
                if(it1 == collision_root_simplex.nodes.end()){
                    mstp1.escape_nodes.push_back(n);
                }
            }
            collision_root_simplex.nodes.erase(mdstp.midpoint_node);
            for(const auto&  p:mstp1.escape_nodes){
                collision_root_simplex.nodes.insert(p);
            }
            multiplicities.push_front(mstp1);
        }
        if(collision_root_simplex.nodes.size() == max_dim){
            break;
        }
    }
}

std::string Simplex::to_string(){
    std::vector<long> node_vector(nodes.begin(),nodes.end());
    std::sort(node_vector.begin(),node_vector.end());
    std::string x ="";
    x += descend_path.to_string();
    return x;
}

Simplex::Simplex(){
    descend_path = HashableGraphPath{};
    nodes = std::unordered_set<long>{};
}

Simplex::Simplex(HashableGraphPath const* _descend_path,const std::unordered_map<long,size_t>& node2ind){
    descend_path = HashableGraphPath{};
    for(const auto& p: *_descend_path){
        descend_path.push_back(p);
    }
    nodes = std::unordered_set<long> {};
    for(const auto& p: node2ind){
        nodes.insert(p.first);
    }
}

Simplex::Simplex(HashableGraphPath const* _descend_path, const std::vector<long>& _nodes){
    descend_path = HashableGraphPath{};
    for(const auto& p: *_descend_path){
        descend_path.push_back(p);
    }
    nodes = std::unordered_set<long> {};
    for(const auto& p: _nodes){
        nodes.insert(p);
    }
}

Simplex::Simplex(size_t max_dim){
    descend_path = HashableGraphPath{};
    nodes = std::unordered_set<long> {};
    for(long i = -max_dim; i < 0; ++i){
        nodes.insert(i);
    }
}
Simplex::Simplex(HashableGraphPath const* _descend_path, size_t max_dim){
    descend_path = HashableGraphPath{};
    nodes = std::unordered_set<long> {};
    for(long i = -max_dim; i < 0; ++i){
        nodes.insert(i);
    }
    for(auto& p: *_descend_path){
        forward(p);
    }
}

Simplex::Simplex(Simplex const*simplex){
    descend_path = HashableGraphPath{};
    for(const auto& p: simplex->descend_path){
        descend_path.push_back(p);
    }
    nodes =std::unordered_set<long> {};
    for(const auto& p: simplex->nodes){
        nodes.insert(p);
    }
}

bool Simplex::not_root() const{
    return !descend_path.empty();
}
MidpointStep Simplex::back_track(){
    auto mstp = descend_path.back();
    nodes.erase(mstp.midpoint_node);
    nodes.insert(mstp.escape_node);
    descend_path.pop_back();
    return mstp;
}
void Simplex::forward(MidpointStep mstp){
    nodes.erase(mstp.escape_node);
    nodes.insert(mstp.midpoint_node);
    descend_path.push_back(mstp);
}

MidpointStepMultiplicity::MidpointStepMultiplicity(long _midpoint_node,size_t _midpoint_alias){
    midpoint_node = _midpoint_node;
    midpoint_alias = _midpoint_alias;
    escape_nodes = std::vector<long>{};
}


SimplexIterator::SimplexIterator(Simplex* _leaf_simplex,
                                MidpointTable* _midpoint_table,
                                Simplex* _collision_root,
                                std::deque<MidpointStepMultiplicity>* multiplicities,
                                size_t _max_alias){
    max_alias = _max_alias;
    leaf_simplex = _leaf_simplex;
    midpoint_table = _midpoint_table;
    collision_root = _collision_root;
    discovered_midpoint_steps = std::unordered_map<size_t,std::pair<MidpointStepMultiplicity,bool>>{};
    multiplicity_counter_stack = std::vector<std::pair<size_t,size_t>>{};
    for(const auto &p: *multiplicities){
        discovered_midpoint_steps.insert({p.midpoint_alias,{p,true}});
    }
    if(multiplicities->empty()){
        return;
    }
    multiplicity_counter_stack.push_back({multiplicities->at(0).midpoint_alias,0});
}

bool SimplexIterator::foward(){
    if(multiplicity_counter_stack.empty()){
        return false;
    }
    auto [midpoint_alias,multiplicity_index] = multiplicity_counter_stack.back();
    auto [midpoint_multiplicity,base_flag] = discovered_midpoint_steps.at(midpoint_alias);
    if(multiplicity_index == 0){
        auto esc_node = midpoint_multiplicity.escape_nodes[multiplicity_index];
        auto midpoint_node = midpoint_multiplicity.midpoint_node;
        leaf_simplex->nodes.erase(esc_node);
        leaf_simplex->nodes.insert(midpoint_node);
        MidpointStep mstp{midpoint_node,midpoint_alias,
            std::numeric_limits<size_t>::max(),esc_node,
            std::numeric_limits<size_t>::max()};
        leaf_simplex->descend_path.push_back(mstp);
        if(base_flag){
            collision_root->nodes.insert(midpoint_node);
            for(const auto & n:midpoint_multiplicity.escape_nodes){
                collision_root->nodes.erase(n);
            }
        }
    }else if(multiplicity_index == midpoint_multiplicity.escape_nodes.size()){
        multiplicity_counter_stack.pop_back();
        leaf_simplex->descend_path.pop_back();
        auto esc_node = midpoint_multiplicity.escape_nodes[multiplicity_index-1];
        auto midpoint_node = midpoint_multiplicity.midpoint_node;
        leaf_simplex->nodes.erase(midpoint_node);
        leaf_simplex->nodes.insert(esc_node);
        if(base_flag){
            collision_root->nodes.erase(midpoint_node);
            for(const auto & n:midpoint_multiplicity.escape_nodes){
                collision_root->nodes.insert(n);
            }
        }else{
            auto midpoint_alias = midpoint_multiplicity.midpoint_alias;
            discovered_midpoint_steps.erase(midpoint_alias);
        }
        return false;
    }else{
        auto prev_esc_node = midpoint_multiplicity.escape_nodes[multiplicity_index-1];
        auto esc_node = midpoint_multiplicity.escape_nodes[multiplicity_index];
        auto midpoint_node = midpoint_multiplicity.midpoint_node;
        leaf_simplex->nodes.erase(esc_node);
        leaf_simplex->nodes.insert(prev_esc_node);
        leaf_simplex->descend_path.pop_back();
        MidpointStep mstp{midpoint_node,midpoint_alias,
            std::numeric_limits<size_t>::max(),esc_node,
            std::numeric_limits<size_t>::max()};
        leaf_simplex->descend_path.push_back(mstp);
    }
    multiplicity_counter_stack.back().second += 1;
    size_t next_midpoint_alias = midpoint_table->find_midpoint_alias(leaf_simplex->nodes, leaf_simplex->descend_path);
    if(next_midpoint_alias == std::numeric_limits<size_t>::max()){
        return true;
    }else if(next_midpoint_alias == max_alias){
        return true;
    }
    if(discovered_midpoint_steps.find(next_midpoint_alias) == discovered_midpoint_steps.end()){
        MidpointStepMultiplicity msm {midpoint_table->alias2node[next_midpoint_alias], next_midpoint_alias};
        auto midpoint = &midpoint_table->midpoints[next_midpoint_alias];
        for(const auto& p: midpoint->nodes){
            if(collision_root->nodes.find(p) == collision_root->nodes.end()){
                msm.escape_nodes.push_back(p);
            }
        }
        discovered_midpoint_steps.insert({
            next_midpoint_alias, {msm,false}
        });
    }
    multiplicity_counter_stack.push_back({next_midpoint_alias,0});
    return false;
}


bool SimplexIterator::depleted(){
    return multiplicity_counter_stack.empty();
}
