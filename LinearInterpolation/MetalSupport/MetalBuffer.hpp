//
//  interface.hpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 6/24/24.
//




#ifndef MetalBuffer_hpp
#define MetalBuffer_hpp

#include "../control_macros.hpp"

#ifndef NO_METAL
#include "Metal/Metal.hpp"
#include <iostream>
#include <stdio.h>

struct MetalBufferPiece{
    MetalBufferPiece(unsigned int _nbytes = 0,unsigned int _byte_offset = 0,void* dPtr = nullptr);
    bool is_allocated() const;
    template <typename T>
    T* content_pointer() const;
    template <typename T>
    unsigned int numel() const;
    void to(MTL::Device* _device);
    bool free();
    void set_buffer(const MetalBufferPiece& x);
    void set_nbytes(unsigned int);
    void set_offset(unsigned int);
    unsigned int get_nbytes() const;
    unsigned int get_byte_offset() const;
    MTL::Buffer* get_buffer() const;
    void* get_data_ptr() const;
    MetalBufferPiece(const MetalBufferPiece& x){
        nbytes = x.get_nbytes();
        byte_offset = x.get_byte_offset();
        data_ptr = x.get_data_ptr();
        buffer = x.get_buffer();
    }
    void move_buffer(MetalBufferPiece& x);
    ~MetalBufferPiece();
private:
    unsigned int nbytes;
    unsigned int byte_offset;
    MTL::Buffer* buffer;
    void* data_ptr;
};


template <typename T>
T* MetalBufferPiece::content_pointer() const{
    if(buffer == nullptr){
        return nullptr;
    }
    char* charPtr = (char*) buffer->contents();
    charPtr += byte_offset/sizeof(char);
    return (T*) charPtr;
}
template <typename T>
unsigned int MetalBufferPiece::numel() const{
    return nbytes/sizeof(T);
}

#endif /*NO_METAL*/
#endif /* MetalBuffer_hpp */

