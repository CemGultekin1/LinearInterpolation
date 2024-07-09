//
//  GlobalMemory.cpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 6/26/24.
//
#include "GlobalMemory.hpp"
#include "../Utils/debug_control.h"


#ifndef NO_METAL


std::pair<unsigned int,unsigned int> BufferSlicer::index_forward(unsigned int n){
    n = roundUpToNearestMultiple(n,grain);
    if(beg + n > end){
        return {beg,beg};
    }else{
        auto oldbeg = beg;
        beg += n;
        return {oldbeg,beg};
    }
}



std::string MetalSparseMidpoint::to_string(){
    return "midpoint_index = " + std::to_string(midpoint_index)  + ", nnz = " + std::to_string(nodes.numel());
}
unsigned int MetalSparseMidpoint::numel() const{
    return nnz;
}



MetalSparseMidpoint::MetalSparseMidpoint(const TypedMetalBufferPiece<float>& mp,float tol,MTL::Device* device,unsigned int mindex)
:midpoint_index(mindex){
    float *dataPtr = mp.content_pointer();
    float _maxel = std::numeric_limits<float>::min();
    for(unsigned int index = 0; index <mp.numel(); index ++){
        _maxel = std::max(_maxel,dataPtr[index]);
    }
    
    float lolim = _maxel*tol;
    
    unsigned int n = 0;
    float sum = 0;
    for(unsigned int index = 0; index <mp.numel(); index ++){
        if(dataPtr[index] > lolim){
            n++;
            sum += dataPtr[index];
        }
    }
    BufferSlicer bfs{};
    auto [nb00,nb01] = bfs.index_forward(n*sizeof(float));
    auto [nb10,nb11] = bfs.index_forward(n*sizeof(int));
    
    if(nb11 == 0){
        return;
    }
    set_nbytes(nb11);
    to(device);
    
    weights.set_nbytes(nb01);
    weights.set_offset(nb00);
    weights.set_buffer(*this);
    
    nodes.set_nbytes(nb11 - nb10);
    nodes.set_offset(nb10);
    nodes.set_buffer(*this);
    
    float* weightsPtr = weights.content_pointer();
    unsigned int* nodesPtr = nodes.content_pointer();
    nnz = 0;
    for(unsigned int index = 0; index <mp.numel(); index ++){
        if(dataPtr[index] > lolim){
            nodesPtr[nnz] = index;
            weightsPtr[nnz] = dataPtr[index]/sum;
            nnz++;
        }
    }
    DEBUG_PRINT("Sparse midpoint: nnz = %d, tol = %f\n",nnz,tol);
}

void fill_with_random(TypedMetalBufferPiece<float>&x,unsigned int dimension,unsigned int offset){
    float *dataPtr = x.content_pointer();
    for(unsigned int index = 0; index <offset; index ++){
        dataPtr[index] = 0;
    }
//
    for (unsigned int index = offset; index < offset+dimension; index++)
    {
        dataPtr[index] = (float) rand() / (float)(RAND_MAX);
    }
    for(unsigned int index = offset+dimension; index < x.numel(); index ++){
        dataPtr[index] = 0;
    }
}




int roundUpToNearestMultiple(int numToRound, int multiple)
{
    assert(multiple);
    return ((numToRound + multiple - 1) / multiple) * multiple;
}

#endif
