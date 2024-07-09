//
//  simplex_tree.hpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 3/31/24.
//

#ifndef graph_descend_hpp
#define graph_descend_hpp


#include <unordered_map>
#include <unordered_set>
#include "graph_path.hpp"
#include "graph_nodes.hpp"


struct MidpointTable{
    std::vector<long> alias2node;
    std::unordered_map<long,std::vector<size_t>> node2aliases;
    std::vector<std::vector<float>> data;
    std::vector<SparsePoint> midpoints;
    std::unordered_map<uint64_t,std::vector<size_t>> hash2midpoint;
    size_t max_dim;
    size_t depth;
    std::unordered_map<size_t,size_t> facing_nodes;
    std::unordered_map<size_t,HashableGraphPath> facing_node_paths;
    MidpointTable();
    long add_node(const std::vector<float>& x);
    void associate(long node,size_t alias);
    size_t add_midpoint(const SparsePoint & p, size_t x_dim,long node);
    size_t find_midpoint_alias(const std::unordered_map<long,size_t>& node2index, const HashableGraphPath & descend) const;
    size_t find_midpoint_alias(const std::unordered_set<long>& node2index, const HashableGraphPath & descend) const;
    void increment_by_weights(std::vector<float>& x, long node,float weight,int power,int num_threads) const;
    bool check_step_consistency(const MidpointStep& ms) const;
    void update_depth(const HashableGraphPath& path);
    void insert_midpoint_on_hash(const uint64_t& hash_value, const size_t& alias);
};


struct TreeDescend{
    PointWithDictionary* data;
    const MidpointTable * midpoint_table;
    HashableGraphPath descend_path;
    TreeDescend(PointWithDictionary& ,const MidpointTable&);
    TreeDescend(const MidpointTable& _m):data(nullptr),midpoint_table(&_m){};
    void descend(int num_threads);
    float determinant() const;
    void follow_midpoint_with_step_discovery(PointWithDictionary *,int num_threads);
};


#endif 
