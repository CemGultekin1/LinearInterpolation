///*
//CPP translation of original Objective-C MetalAdder.h. Some stuff has been moved over
//here from the cpp file. Source: https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu?language=objc
//
//Original distribution license: LICENSE-original.txt.
//
//Abstract:
//A class to manage all of the Metal objects this app creates.
//*/
//#pragma once
//
//#include "Foundation/Foundation.hpp"
//#include "Metal/Metal.hpp"
//
//
//
//
//class MetalOperation{
//public:    
//    unsigned int _per_thread_compute;
//    unsigned int _num_threadgroups;
//    unsigned int _grid_size;
//    unsigned int _threadgroup_size;
//    unsigned int _per_thread_memory;
//    unsigned int _per_threadgroup_memory;
//    std::string _kernel_name;
//    MTL::Device *_mDevice;
//    MTL::CommandQueue *_mCommandQueue;
//    MTL::ComputePipelineState *_pso;
//    MetalOperation(MTL::Device *device,unsigned int per_thread_compute=1);
//    virtual ~MetalOperation();
//    void send_compute_command(MetalMemory* mm);
//    virtual unsigned int configure(unsigned int input_size);
//    virtual std::string get_fxn_name();
//    virtual void setup_pso();
//    virtual void setup_inputs(MetalMemory* mm,MTL::ComputeCommandEncoder *computeEncoder) = 0;
//    virtual void encode_add_command(MetalMemory* mm,MTL::ComputeCommandEncoder *computeEncoder);
//    virtual bool verify_computation(MetalMemory* mm ) = 0;
//};
//
//class FirstLayerOperator: public MetalOperation{
//public:
////    std::string _kernel_name =
//    virtual void setup_inputs(MetalMemory* mm,MTL::ComputeCommandEncoder *computeEncoder) override;
//    virtual bool verify_computation(MetalMemory* mm ) override;
//};
//
