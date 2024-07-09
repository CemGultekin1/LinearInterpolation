//
//  GlobalMemory.hpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 5/20/24.
//

#ifndef GlobalMemory_hpp
#define GlobalMemory_hpp
#include "../control_macros.hpp"

#ifndef NO_METAL
#include "init_metal_header.hpp"
#include "MetalBuffer.hpp"



template <typename T>
struct TypedMetalBufferPiece:public MetalBufferPiece{
    TypedMetalBufferPiece(unsigned int _numel = 0):MetalBufferPiece(_numel*sizeof(T)){};
    T* content_pointer() const{
        return MetalBufferPiece::content_pointer<T>();
    }
    unsigned int numel() const{
        return MetalBufferPiece::numel<T>();
    }
    MetalBufferPiece to_buffer_piece() const;
};


template <typename T>
MetalBufferPiece TypedMetalBufferPiece<T>::to_buffer_piece() const{
    MetalBufferPiece x{get_nbytes(),get_byte_offset()};
    x.set_buffer(*this);
    return x;
}


struct BufferSlicer{
    unsigned int beg;
    unsigned int end;
    unsigned int grain;
    BufferSlicer(unsigned int _beg = 0,
                 unsigned int _end = std::numeric_limits<unsigned int>::max(),
                 unsigned int _grain = BANK_SIZE_IN_BYTES):beg(_beg),end(_end),grain(_grain){};
    std::pair<unsigned int,unsigned int> index_forward(unsigned int n);
};

struct MetalBufferSlicer:public BufferSlicer{
private:
    const MetalBufferPiece base_buffer;
public:
    MetalBufferSlicer(const MetalBufferPiece& x,unsigned int _grain = BANK_SIZE_IN_BYTES):BufferSlicer(0, x.get_nbytes(), _grain),base_buffer(x){};
    template <typename T>
    MetalBufferPiece step_forward(unsigned int n = 1);
};


template <typename T>
MetalBufferPiece MetalBufferSlicer::step_forward(unsigned int n){
    auto [x,y] = index_forward(sizeof(T)*n);
    if(x==y){
        MetalBufferPiece mbp{};
        return mbp;
    }else{
        MetalBufferPiece mbp{y - x,x};
        mbp.set_buffer(base_buffer);
        return mbp;
    }
}



struct MetalSparseMidpoint:public MetalBufferPiece{
    TypedMetalBufferPiece<unsigned int> nodes;
    TypedMetalBufferPiece<float> weights;
    unsigned int nnz;
    unsigned int midpoint_index;
    float rtol;
    MetalSparseMidpoint(const TypedMetalBufferPiece<float>&,float rtol,MTL::Device* device,unsigned int midpoint_index);
    std::string to_string();
    unsigned int numel() const;
};

void fill_with_random(TypedMetalBufferPiece<float>&x,unsigned int dimension,unsigned int offset);

int roundUpToNearestMultiple(int numToRound, int multiple);






#endif /* NO_METAL */
#endif /* GlobalMemory_hpp */

