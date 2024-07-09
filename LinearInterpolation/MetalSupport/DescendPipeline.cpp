////
////  DescendPipeline.cpp
////  LinearInterpolation
////
////  Created by Cem Gultekin on 6/19/24.
////
//
//#include "DescendPipeline.hpp"
//
////class DescendPipeline{
////public:
////    MTL::Device *device;
////    MetalCommandQueue * mQueue;
////    MetalMemoryManager * gMem;
////    unsigned int per_thread;
////    
////    FirstLayerKernel* flk;
////    FindExitKernel* fek;
////    MidpointOperationKernel* mok;
////    
////    MetalBufferPiece* temp;
////    
////    MetalPoint* input;
////    
////    DescendPipeline(MTL::Device* device,
////                    MetalCommandQueue* mQueue,
////                    MetalMemoryManager* gMem,
////                    unsigned int per_thread):device(device),mQueue(mQueue),gMem(gMem){};
////    
////    MetalBufferPiece* sufficient_compute_memory(unsigned int nbytes);
////    void operator()(std::vector<float>&);
////};
////void DescendPipeline::operator()(std::vector<float>& x){
////    gMem->free_memory(<#MetalBuffer &#>)
////    MetalPoint(x.size()*sizeof(float),x.size()+1,(void*)&x[0]);
////}
