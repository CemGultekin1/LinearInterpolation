//
//  MetalBuffer.cpp
//  LinearInterpolation

//  Created by Cem Gultekin on 6/26/24.
//
#include "MetalBuffer.hpp"

#ifndef NO_METAL
MetalBufferPiece::MetalBufferPiece(unsigned int _nbytes,unsigned int _byte_offset, void* dPtr ){
    buffer = nullptr;
    nbytes = _nbytes;
    byte_offset = _byte_offset;
    data_ptr = dPtr;
}


bool MetalBufferPiece::is_allocated() const{
    return buffer != nullptr;
}



bool MetalBufferPiece::free(){
    if(buffer != nullptr){
        return false;
    }
    buffer->release();
    buffer = nullptr;
    return true;
}

void MetalBufferPiece::to(MTL::Device *_device){
    free();
    if(data_ptr != nullptr){
        buffer = _device->newBuffer(data_ptr, nbytes, MTL::ResourceStorageModeManaged);
    }else{
        buffer = _device->newBuffer(nbytes, MTL::ResourceStorageModeShared);
    }
    
}




void MetalBufferPiece::set_buffer(const MetalBufferPiece& x){
    free();
    buffer = x.get_buffer();
}
MTL::Buffer* MetalBufferPiece::get_buffer() const{
    return buffer;
}


unsigned int MetalBufferPiece::get_nbytes() const{
    return nbytes;
}
unsigned int MetalBufferPiece::get_byte_offset() const{
    return byte_offset;
}


void* MetalBufferPiece::get_data_ptr() const{
    return data_ptr;
}



MetalBufferPiece::~MetalBufferPiece(){
    free();
    data_ptr = nullptr;
}

void MetalBufferPiece::set_nbytes(unsigned int x){
    nbytes = x;
}


void MetalBufferPiece::set_offset(unsigned int x){
    byte_offset = x;
}


void MetalBufferPiece::move_buffer(MetalBufferPiece& x){
    x.set_buffer(*this);
    this->buffer = nullptr;
}
#endif
