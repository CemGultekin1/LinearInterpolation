//
//  KernelLibrary.hpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 5/19/24.
//

#ifndef KernelLibrary_hpp
#define KernelLibrary_hpp

#include <stdio.h>
#include <string>
#include "GlobalMemory.hpp"
#include "CommandQueue.hpp"


class KernelFxn{
public:
    std::string  _kernel_root;
    std::string  _unique_kernel_name;
    unsigned int _per_thread_threadgroup_memory;
    unsigned int _thread_compute;
    
    unsigned int _grid_size;
    unsigned int _threadgroup_size;
    unsigned int _num_threadgroups;
    bool _cpu_verify_flag;
    MetalCommandQueue* _mcq;
    MTL::Device* _mDevice;
    MTL::ComputePipelineState* _compute_pipeline_state_ptr;
    KernelFxn(std::string  kernel_root,unsigned int per_thread_threadgroup_memory,unsigned int per_thread_compute,MetalCommandQueue *mCommandQueu,bool cpu_verify_flag);
    std::pair<MTL::CommandBuffer*, MTL::ComputeCommandEncoder*> create_encoder_buffer();
    void free_encoder_buffer(MTL::CommandBuffer*, MTL::ComputeCommandEncoder*);
    virtual void encode_arguments(MTL::ComputeCommandEncoder*) = 0;
    virtual void create_threadgroup_memory(MTL::ComputeCommandEncoder* encoder);
    virtual unsigned int configure(unsigned int input_size);
};


class FirstLayerKernel:public KernelFxn{
public:
    FirstLayerKernel(unsigned int per_thread_compute,MetalCommandQueue *mCommandQueu,bool cpu_verify = false):KernelFxn("first_layer",sizeof(float),per_thread_compute, mCommandQueu,cpu_verify){};
    void encode_arguments(MTL::ComputeCommandEncoder*,SizedMetalBuffer<float>& input, SizedMetalBuffer<float>& temp);
    virtual void encode_arguments(MTL::ComputeCommandEncoder*){};
    SizedMetalBuffer<float> memory_request();
    void cpu_operation(SizedMetalBuffer<float>&  in,float*out);
    bool operator()(SizedMetalBuffer<float>& input, SizedMetalBuffer<float>& temp);
};



class FindExitKernel:public KernelFxn{
public:
    FindExitKernel(unsigned int per_thread_compute,MetalCommandQueue *mCommandQueu,bool cpu_verify = false):KernelFxn("find_midpoint_exit_node",sizeof(float) +sizeof(unsigned int)+sizeof(char),per_thread_compute, mCommandQueu,cpu_verify){};
    void encode_arguments(MTL::ComputeCommandEncoder*,
                          SizedMetalBuffer<float>& input,
                          Midpoint&,
                          SizedMetalBuffer<unsigned int>&,
                          SizedMetalBuffer<float>&,
                          SizedMetalBuffer<char>&);
    virtual void encode_arguments(MTL::ComputeCommandEncoder*) override {};
    std::tuple<SizedMetalBuffer<unsigned int>,SizedMetalBuffer<float>,SizedMetalBuffer<char>>
    memory_request();
    std::tuple<bool,unsigned int,float> cpu_operation(SizedMetalBuffer<float>& in,unsigned int*midpoint_nodes,float*midpoint_weights,unsigned int);
    std::tuple<bool,unsigned int,float>  operator()(SizedMetalBuffer<float>&,Midpoint&,SizedMetalBuffer<unsigned int>&,
                    SizedMetalBuffer<float>&,SizedMetalBuffer<char>&);
    void create_threadgroup_memory(MTL::ComputeCommandEncoder* encoder) override;
};


class MidpointOperationKernel:public KernelFxn{
public:
    MidpointOperationKernel(unsigned int per_thread_compute,MetalCommandQueue *mCommandQueu,bool cpu_verify = false):KernelFxn("midpoint_operation",0,per_thread_compute, mCommandQueu,cpu_verify){};
    void encode_arguments(MTL::ComputeCommandEncoder*,
                          SizedMetalBuffer<float>& input,
                          const Midpoint&,
                          const float&,
                          const unsigned int&,
                          const unsigned int&);
    virtual void encode_arguments(MTL::ComputeCommandEncoder*) override {};
    void cpu_operation(
                    SizedMetalBuffer<float>& ,const Midpoint& ,
                    std::vector<float>&,const unsigned int& ,
                    const float& ,const unsigned int& );
    void operator()(SizedMetalBuffer<float>&,
                    const Midpoint& ,
                    const unsigned int&,
                    const float&,
                    const unsigned int&);
};




#endif /* KernelLibrary_hpp */
