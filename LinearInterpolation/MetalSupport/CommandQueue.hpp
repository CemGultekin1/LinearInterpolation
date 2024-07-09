//
//  CommandQueue.hpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 5/19/24.
//



#ifndef CommandQueue_hpp
#define CommandQueue_hpp

#include "../control_macros.hpp"
#ifndef NO_METAL



#include <stdio.h>
#include <iostream>

#include <Metal/Metal.hpp>



class MetalCommandQueue{
public:
    MTL::Device *_mDevice;
    MTL::CommandQueue *_mCommandQueue;
    MetalCommandQueue(MTL::Device *device);
    MTL::CommandBuffer * create_buffer();
    ~MetalCommandQueue();
};
#endif


#endif /* CommandQueue_hpp */

