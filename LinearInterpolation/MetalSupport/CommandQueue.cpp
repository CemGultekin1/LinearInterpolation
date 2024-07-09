////
////  CommandQueue.cpp
////  LinearInterpolation
////
////  Created by Cem Gultekin on 5/19/24.
////
//
//
#include "CommandQueue.hpp"

#ifndef NO_METAL

MetalCommandQueue::MetalCommandQueue(MTL::Device *device){
    _mDevice = device;
    _mCommandQueue = _mDevice->newCommandQueue();
    if (_mCommandQueue == nullptr)
    {
        std::cout << "Failed to find the command queue." << std::endl;
        return;
    }
    
}


MTL::CommandBuffer * MetalCommandQueue::create_buffer(){
    MTL::CommandBuffer *commandBuffer = _mCommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);
    return commandBuffer;
};



MetalCommandQueue::~MetalCommandQueue(){
    _mCommandQueue->release();
}

#endif
