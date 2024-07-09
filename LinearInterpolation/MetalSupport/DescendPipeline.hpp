////
////  DescendPipeline.hpp
////  LinearInterpolation
////
////  Created by Cem Gultekin on 6/19/24.
////
//
//#ifndef DescendPipeline_hpp
//#define DescendPipeline_hpp
//
//#include <stdio.h>
//#include "KernelLibrary.hpp"
//
//
//
//class DescendPipeline{
//public:
//    MTL::Device *device;
//    MetalCommandQueue * mQueue;
//    unsigned int per_thread;
//    
//    FirstLayerKernel* flk;
//    FindExitKernel* fek;
//    MidpointOperationKernel* mok;
//    
//    MetalBufferPiece* temp;
//    
//    MetalPoint* input;
//    
//    DescendPipeline(MTL::Device* device,
//                    MetalCommandQueue* mQueue,
//                    unsigned int per_thread):device(device),mQueue(mQueue){};
//    
//    MetalBufferPiece* sufficient_compute_memory(unsigned int nbytes);
//    void operator()(std::vector<float>&);
//};
//
//#endif /* DescendPipeline_hpp */
