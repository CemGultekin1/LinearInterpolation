//
//  GlobalMemory.cpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 5/20/24.
//

#include "GlobalMemory.hpp"
#include <iostream>


GlobalMetalMemory::GlobalMetalMemory(MTL::Device * device){
    _mDevice = device;
    _memory_address_counter = 0;
}







void GlobalMetalMemory::fill_with_random(SizedMetalBuffer<float>& x,unsigned int dimension,unsigned int offset){
    float *dataPtr = (float *)x.buffer->contents();
    for(unsigned int index = 0; index <offset; index ++){
        dataPtr[index] = 0;
    }
    
    for (unsigned int index = offset; index < offset+dimension; index++)
    {
        dataPtr[index] = (float) rand() / (float)(RAND_MAX);
    }
    for(unsigned int index = offset+dimension; index < x.size; index ++){
        dataPtr[index] = 0;
    }
}


GlobalMetalMemory::~GlobalMetalMemory()
{
    for(auto [p,x]: _full_memory){
        x->release();
    }
}

void GlobalMetalMemory::add_midpoint(SizedMetalBuffer<float>& x,float tolerance){
    float *dataPtr = (float *)x.buffer->contents();
    float _maxel = std::numeric_limits<float>::min();
    for(unsigned int index = 0; index <x.size; index ++){
        _maxel = std::max(_maxel,dataPtr[index]);
    }
    
    float lolim = _maxel*tolerance;
    
    
    unsigned int n = 0;
    float sum = 0;
    for(unsigned int index = 0; index <x.size; index ++){
        if(dataPtr[index] > lolim){
            n++;
            sum += dataPtr[index];
        }
    }
    

    SizedMetalBuffer<float> weights(n);
    SizedMetalBuffer<int> nodes(n);
    reserve_gpu_memory(weights);
    reserve_gpu_memory(nodes);
    float* weightsPtr = (float*) weights.buffer->contents();
    int* nodesPtr = (int*) nodes.buffer->contents();
    n = 0;
    for(unsigned int index = 0; index <x.size; index ++){
        if(dataPtr[index] > lolim){
            nodesPtr[n] = index - x.origin;
            weightsPtr[n] = dataPtr[index]/sum;
            n++;
        }
    }
    Midpoint m{nodes,weights};
    m.midpoint_index = static_cast<unsigned int>(_midpoints.size());
    _midpoints.emplace_back(m);
}
