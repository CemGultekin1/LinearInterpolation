//
//  CommandQueue.hpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 5/19/24.
//

#ifndef CommandQueue_hpp
#define CommandQueue_hpp


#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include <stdio.h>
#include <iostream>



class MetalCommandQueue{
public:
    MTL::Device *_mDevice;
    MTL::CommandQueue *_mCommandQueue;
    MetalCommandQueue(MTL::Device *device);
    MTL::CommandBuffer * create_buffer();
    ~MetalCommandQueue();
};

#endif /* CommandQueue_hpp */

