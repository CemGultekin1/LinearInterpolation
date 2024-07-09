//
//  path_hash.hpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 4/3/24.
//

#ifndef graph_path_hpp
#define graph_path_hpp

#include <stdio.h>
#include <cstdint>
#include <bit>
#include <vector>


struct MidpointStep {
    size_t midpoint_alias;
    long midpoint_node;
    size_t escape_index;
    size_t escape_alias;
    long escape_node;
    MidpointStep(
       long midpoint_node,
    size_t midpoint_alias,
      size_t escape_index,
        long escape_node,
       size_t escape_alias):
            midpoint_alias(midpoint_alias),
            midpoint_node(midpoint_node),
            escape_index(escape_index),
            escape_alias(escape_alias),
            escape_node(escape_node){};
    MidpointStep();
    std::string to_string() const;
    void set_escape_node(long new_escape_node);
};




void hash_forward(uint64_t& hash,long x);
void hash_backward(uint64_t& hash,long x);

class HashableGraphPath: public std::vector<MidpointStep>{
private:
    uint64_t hash_value;
public:
    HashableGraphPath();
    HashableGraphPath(const HashableGraphPath&);
    void push_back(MidpointStep escape_midpoint);
    void pop_back();
    uint64_t hash() const;
    std::string to_string() const;
};

#endif /* graph_path_hpp */
