//
//  GlobalMemory.hpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 5/20/24.
//

#ifndef GlobalMemory_hpp
#define GlobalMemory_hpp

#include <stdio.h>
#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"

template <typename T>
struct SizedMetalBuffer{
    unsigned int nbytes;
    unsigned int sizeof_type;
    unsigned int size;
    unsigned int global_memory_address;
    unsigned int origin;
    MTL::Buffer* buffer;
    SizedMetalBuffer<T>(unsigned int _size,unsigned int _origin = 0){
        size = _size;
        nbytes = sizeof(T)*size;
        sizeof_type = sizeof(T);
        buffer = nullptr;
        origin = _origin;
    }
    bool is_allocated() const{
        return buffer != nullptr;
    }
};

struct Midpoint{
    SizedMetalBuffer<int> nodes;
    SizedMetalBuffer<float>  weights;
    unsigned int midpoint_index;
    Midpoint(SizedMetalBuffer<int>x,SizedMetalBuffer<float>y):nodes(x),weights(y){};
    std::string to_string(){
        return "midpoint_index = " + std::to_string(midpoint_index)  + ", nnz = " + std::to_string(nodes.size);
    }
};


class GlobalMetalMemory{
public:
    MTL::Device *_mDevice;
    std::vector<Midpoint> _midpoints;
    std::unordered_map<unsigned int, MTL::Buffer*> _full_memory;
    unsigned int _memory_address_counter;
    GlobalMetalMemory(MTL::Device * device);
    template <typename T>
    void reserve_gpu_memory(SizedMetalBuffer<T>&);
    template <typename T>
    void free_memory(SizedMetalBuffer<T>&);
    void fill_with_random(SizedMetalBuffer<float>&,unsigned int dimension,unsigned int offset = 1);
    void add_midpoint(SizedMetalBuffer<float>& x,float tolerance = 0.001);
    ~GlobalMetalMemory();
};


template <typename T>
void GlobalMetalMemory::reserve_gpu_memory(SizedMetalBuffer<T>& x)
{
    if(x.is_allocated()){
        assert(_full_memory.find(x.global_memory_address) != _full_memory.end());
        free_memory(x);
    }
    x.buffer = _mDevice->newBuffer(x.nbytes, MTL::ResourceStorageModeShared);
    x.global_memory_address = _memory_address_counter;
    _memory_address_counter += 1;
    _full_memory[x.global_memory_address] = x.buffer;
}


template <typename T>
void GlobalMetalMemory::free_memory(SizedMetalBuffer<T>&x){
    if(_full_memory.find(x.global_memory_address) != _full_memory.end()) {
        _full_memory[x.global_memory_address]->release();
        _full_memory.erase(x.global_memory_address);
    }
}

#endif /* GlobalMemory_hpp */
