//
//  facing_node.cpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 4/5/24.
//

#include "facing_node.hpp"
#include <assert.h>


size_t FacingNodes::HashFunction::operator()(const FacingNodes &pos) const{
    auto n1 =pos.node1;
    auto n2 =pos.node2;
    if(n1 < n2 ){
        std::swap(n1,n2);
    }
    size_t rowHash = std::hash<long>()(n1);
    size_t colHash = std::hash<long>()(n2) << 1;
    return rowHash ^ colHash;
};
std::string FacingNodes::to_string() const{
    return "(" + std::to_string(node1) + "," + std::to_string(node2) +") cost = " + std::to_string(this->resim_cost());
}
float FacingNodes::resim_cost() const{
    if(convex_flag){
        return posterior_resim_cost - prior_resim_cost;
    }else{
        return std::numeric_limits<float>::max();
    }
    
}

bool FacingNodes::operator==(const FacingNodes &fn) const{
    auto n1 = fn.node1;
    auto n2 = fn.node2;
    if(n1 < n2){
        std::swap(n1,n2);
    }
    auto n1_ = this->node1;
    auto n2_ = this->node2;
    if(n1_ < n2_){
        std::swap(n1_,n2_);
    }
    return n1 == n1_ && n2 == n2_;    
}

FacingNodeFinder::FacingNodeFinder(const MidpointTable& _midpoint_table,
                                           const Simplex& _simplex,
                                           size_t _final_midpoint_alias){
    simplex = Simplex{&_simplex};
    org_simplex = Simplex{&_simplex};
    midpoint_table = _midpoint_table;
    final_midpoint_alias = _final_midpoint_alias;
    collision_trace = std::unordered_map<size_t,MidpointStep>{};
    remaining_nodes = std::unordered_map<long,size_t>{};
    final_midpoint_node = midpoint_table.alias2node[final_midpoint_alias];
//    std::cout << "rank = " << midpoint_table.midpoints[final_midpoint_alias].nodes->size() << ": ";
    size_t i = 0;
    for(auto p: midpoint_table.midpoints[final_midpoint_alias].nodes){
//        std::cout << p << ",";
        remaining_nodes.insert({p,i});
        ++i;
    }
//    std::cout << "\n simplex = \n";
//    for(auto node : _simplex.nodes){
//        std::cout << node << ",";
//    }
//    
//    std::cout << "\n";
//    
    level_remaining_nodes = std::vector<std::pair<long,size_t>>{};
    last_level = _simplex.descend_path.size();
}



bool FacingNodeFinder::empty() const{
    return remaining_nodes.empty() || last_level == 0;
}

void FacingNodeFinder::discover_all(std::vector<FacingNodes>& facing_nodes_list){
    while(!empty()){
        ascend(facing_nodes_list);
    }
}

void FacingNodeFinder::descend(MidpointStep& mstp,long &n){
    simplex.forward(mstp);
    auto midpoint_alias = midpoint_table.find_midpoint_alias(simplex.nodes, simplex.descend_path);
    while(midpoint_alias != std::numeric_limits<size_t>::max() && midpoint_alias != final_midpoint_alias){
        if(collision_trace.find(midpoint_alias) != collision_trace.end()){
            simplex.forward(collision_trace[midpoint_alias]);
        }else{
            auto midpoint_node = midpoint_table.alias2node[midpoint_alias];
            MidpointStep mstp1(midpoint_node, midpoint_alias, std::numeric_limits<size_t>::max(), n, std::numeric_limits<size_t>::max());
            simplex.forward(mstp1);
            n = midpoint_node;
        }
        midpoint_alias = midpoint_table.find_midpoint_alias(simplex.nodes, simplex.descend_path);
    }
}

void FacingNodeFinder::org_simplex_forward(size_t node_index){
    MidpointStep mstp1{};
    mstp1.midpoint_node = final_midpoint_node;
    mstp1.midpoint_alias = final_midpoint_alias;
    mstp1.escape_node = midpoint_table.midpoints[final_midpoint_alias].nodes.at(node_index);
    org_simplex.forward(mstp1);
}
void FacingNodeFinder::org_simplex_backward(){
    org_simplex.back_track();
}

void FacingNodeFinder::push(std::vector<FacingNodes> & fnvec, long n, size_t node_index){
    org_simplex_forward(node_index);
    FacingNodes fn;
    fn.path1 = org_simplex.descend_path;
    fn.path2 = simplex.descend_path;
    fn.node1 = final_midpoint_node;
    fn.node2 = n;
    fn.convex_flag = false;
    fn.prior_resim_cost = 0;//std::numeric_limits<float>::max();
    fn.posterior_resim_cost = 0;//std::numeric_limits<float>::min();
    fnvec.push_back(fn);
    org_simplex_backward();
    if(fn.node1 == fn.node2){
        std::cout << fn.path1.to_string() << "\n";
        std::cout << fn.path2.to_string() << "\n";
        throw std::invalid_argument("");
    }
    while(simplex.descend_path.size() > last_level){
        simplex.back_track();
    }
}
void FacingNodeFinder::ascend(std::vector<FacingNodes>& fnvec){
    while(simplex.descend_path.size() > last_level){
        simplex.back_track();
    }
    if(simplex.descend_path.empty()){
        return;
    }
    auto mstp = simplex.back_track();
    if(remaining_nodes.find(mstp.midpoint_node) != remaining_nodes.end()){
        remaining_nodes.insert({mstp.escape_node,remaining_nodes[mstp.midpoint_node]});
        remaining_nodes.erase(mstp.midpoint_node);
    }
    --last_level;
    level_remaining_nodes.clear();
    collision_trace.insert({mstp.midpoint_alias, mstp});
    auto midpoint = &midpoint_table.midpoints[mstp.midpoint_alias];
    for(size_t i = 0;i < midpoint->nodes.size(); ++i){
        auto n = midpoint->nodes.at(i);
        if(n == mstp.escape_node){
            continue;
        }
        if(remaining_nodes.find(n) == remaining_nodes.end()){
            continue;
        }
        if(midpoint->weights.at(i) < 0){
            /*
             {x,y,F\e} looking for facing node of x
             {x,F} is the mother simplex
             the midpoint is y, escape node is e
             instead we want to go to {y,F}
             facing_alias = x
             mstp.midpoint_alias = y
             */
            
            
            auto facing_alias = midpoint_table.facing_nodes[mstp.midpoint_alias];
            auto path = &midpoint_table.facing_node_paths[facing_alias];
            Simplex sm{path, midpoint_table.max_dim};
            FacingNodeFinder new_ffna{midpoint_table, sm, facing_alias};
            auto iloc = new_ffna.remaining_nodes[mstp.escape_node];
            new_ffna.remaining_nodes.clear();
            new_ffna.remaining_nodes.insert({mstp.escape_node,iloc});
            new_ffna.ascend(fnvec);
            if(fnvec.empty()){
                continue;
            }
            org_simplex_forward(remaining_nodes[n]);
            fnvec.back().path1 = org_simplex.descend_path;
            fnvec.back().node1 = final_midpoint_node;
            org_simplex_backward();
            
        }else{
            level_remaining_nodes.push_back({n,remaining_nodes[n]});
        }
        remaining_nodes.erase(n);
        
    }
    auto org_escape = mstp.escape_node;
    while(!level_remaining_nodes.empty()){
        auto  [n,node_index] = level_remaining_nodes.back();
        level_remaining_nodes.pop_back();
        mstp.set_escape_node(n);
        n = org_escape;
        descend(mstp, n);
        push(fnvec, n, node_index);
    }
    return;
}



bool facing_node_consistency(const FacingNodes& fn, const MidpointTable& midpoint_table){
    Simplex sm1(&fn.path1, midpoint_table.max_dim);
    Simplex sm2(&fn.path2, midpoint_table.max_dim);
    
    bool flag11 = sm1.nodes.find(fn.node1) != sm1.nodes.end();
    bool flag12 = sm1.nodes.find(fn.node2) == sm1.nodes.end();
    bool flag21 = sm2.nodes.find(fn.node1) == sm2.nodes.end();
    bool flag22 = sm2.nodes.find(fn.node2) != sm2.nodes.end();
            
    return flag11 && flag12 && flag21 && flag22;
    
}
