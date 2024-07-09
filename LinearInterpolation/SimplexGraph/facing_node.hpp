//
//  facing_node.hpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 4/5/24.
//

#ifndef facing_node_hpp
#define facing_node_hpp

#include <stdio.h>
#include "graph_descend.hpp"
#include "collisions.hpp"
#include "chrono_methods.hpp"

struct FacingNodes{
    long node1;
    long node2;
    HashableGraphPath path1;
    HashableGraphPath path2;
    float prior_resim_cost;
    float posterior_resim_cost;
    bool convex_flag;
    std::string to_string() const;
    float resim_cost() const;
    bool operator==(const FacingNodes& fn) const;
    struct HashFunction
    {
        size_t operator()(const FacingNodes& pos) const;
    };
};


struct facing_node_comparison
{
    bool ls_flag;
    facing_node_comparison(bool ls):ls_flag(ls){};
    inline bool operator() (const FacingNodes& struct1, const FacingNodes& struct2)
    {
        if(ls_flag){
            return struct1.resim_cost() < struct2.resim_cost();
        }else{
            return struct1.resim_cost() > struct2.resim_cost();
        }
        
    };
};

bool facing_node_consistency(const FacingNodes& fn, const MidpointTable&);

struct FacingNodeFinder{
    MidpointTable midpoint_table;
    std::unordered_map<size_t,MidpointStep> collision_trace;
    std::unordered_map<long,size_t> remaining_nodes;
    std::vector<std::pair<long,size_t>> level_remaining_nodes;
    Simplex simplex;
    Simplex org_simplex;
    size_t last_level;
    size_t final_midpoint_alias;
    long final_midpoint_node;
    FacingNodeFinder(const MidpointTable& midpoint_table,
                         const Simplex& simplex,size_t final_midpoint_alias);
    bool empty() const;
    void org_simplex_forward(size_t node_index);
    void org_simplex_backward();
    void ascend(std::vector<FacingNodes>&);
    void discover_all(std::vector<FacingNodes>&);
    void descend(MidpointStep& mstp,long &n);
    void push(std::vector<FacingNodes>&,long,size_t);
};



#endif /* facing_node_hpp */
