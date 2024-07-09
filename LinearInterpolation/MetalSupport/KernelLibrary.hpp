//
//  KernelLibrary.hpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 5/19/24.
//
#ifndef NO_METAL

#ifndef KernelLibrary_hpp
#define KernelLibrary_hpp

#include "../Utils/debug_control.h"
#include "../Utils/string_manip.hpp"


#include "shaders/macros.h"

#include <string>

#include "CommandQueue.hpp"
#include "GlobalMemory.hpp"

#include "../Utils/formulas.h"


struct KernelConfiguration{
    MTL::Device* device;
    unsigned int grain_size;
    unsigned int per_thread_threadgroup_bytes;
    unsigned int threadgroup_bytes;
    unsigned int temporary_bytes;
    unsigned int grid_size;
    unsigned int threadgroup_size;
    unsigned int num_threadgroups;
    unsigned int padded_dim;
    unsigned int dim;
    unsigned int warpsize;
    float fpdim;
    KernelConfiguration(MTL::Device* device,
                        unsigned int dim,
                        unsigned int grain_size,
                        unsigned int per_thread_threadgroup_bytes = 0,
                        unsigned int warpsize = 32);
    void set_dim(unsigned int new_dim);
    unsigned int rounded_up_dimension(unsigned int new_dim);
    std::string to_string() const;
};


class KernelFxn{
public:
    std::string  _kernel_root;
    std::string  _unique_kernel_name;
    KernelConfiguration* conf;
    MetalCommandQueue* _mcq;
    MTL::Device* _mDevice;
    MTL::ComputePipelineState* _compute_pipeline_state_ptr;
    KernelFxn(std::string  kernel_root,unsigned int per_thread_threadgroup_memory,unsigned int per_thread_compute,MetalCommandQueue *mCommandQueu);
    std::pair<MTL::CommandBuffer*, MTL::ComputeCommandEncoder*> create_encoder_buffer() const;
    void free_encoder_buffer(MTL::CommandBuffer*, MTL::ComputeCommandEncoder*) const;
    virtual void create_threadgroup_memory(MTL::ComputeCommandEncoder* encoder) const;
    unsigned int needed_temporary_bytes(unsigned int dim) const;
    std::string to_string() const{
        return _unique_kernel_name + " = " + conf->to_string() ;
    }
    bool single_threadblock() const{
        return conf->num_threadgroups == 1;
    }
};

class InitializationLayerKernel: public KernelFxn{
public:
    InitializationLayerKernel(std::string  kernel_root,
                    unsigned int per_thread_threadgroup_memory,
                    unsigned int per_thread_compute,
                     MetalCommandQueue *mCommandQueue,
                     unsigned int _dimension,
                     float _mean = 0.5,
                     float _min = 0,
                     float _max = 1):
                    KernelFxn(kernel_root,per_thread_threadgroup_memory,
                    per_thread_compute, mCommandQueue){
                        conf->set_dim(_dimension);
                        min = _min;
                        max = _max;
                        mean = _mean;
                    };
    float min;
    float max;
    float mean;
};


class FirstLayerKernel: public InitializationLayerKernel{
public:
    FirstLayerKernel(unsigned int per_thread_compute,
                     MetalCommandQueue *mCommandQueue,
                     unsigned int _dimension,
                     float _mean = 0.5,
                     float _min = 0,
                     float _max = 1):
                    InitializationLayerKernel(xSTRINGIFY(FIRST_LAYER_KRNL_NM),sizeof(float),
                                              per_thread_compute, mCommandQueue,_dimension, _mean,_min,_max){};
    void encode_arguments(MTL::ComputeCommandEncoder*,
                        TypedMetalBufferPiece<float>& input,
                        MetalBufferPiece& temp) const;
    float cpu_operation(TypedMetalBufferPiece<float>&  in,
                       float*out) const;
    float operator()(TypedMetalBufferPiece<float>& input,
                    MetalBufferPiece& temp) const;
};


class PosteriorFirstLayerKernel: public InitializationLayerKernel{
public:
    PosteriorFirstLayerKernel(unsigned int per_thread_compute,
                     MetalCommandQueue *mCommandQueue,
                     unsigned int _dimension,
                     float _mean = 0.5,
                     float _min = 0,
                     float _max = 1):
                    InitializationLayerKernel(xSTRINGIFY(FIRST_LAYER_PSTR_KRNL_NM),0,
                                        per_thread_compute, mCommandQueue,_dimension,_mean,_min,_max){};
    void encode_arguments(MTL::ComputeCommandEncoder*,
                        TypedMetalBufferPiece<float>& input,float & mean) const;
    void cpu_operation(TypedMetalBufferPiece<float>&  in, float & mean, float*out) const;
    void operator()(TypedMetalBufferPiece<float>& input, float & mean) const;
};


class FindExitKernel:public KernelFxn{
public:
    FindExitKernel(unsigned int per_thread_compute,MetalCommandQueue *mCommandQueu):KernelFxn(xSTRINGIFY(FIND_EXIT_KRNL_NM),sizeof(float) +sizeof(unsigned int)+sizeof(char),per_thread_compute, mCommandQueu){};

    void encode_arguments(MTL::ComputeCommandEncoder*,TypedMetalBufferPiece<float>& input,
                                  MetalSparseMidpoint& midpoint,
                                  MetalBufferPiece& glbl_min_indexes,
                                  MetalBufferPiece& glbl_min_weights);
    std::tuple<unsigned int,float> cpu_operation(TypedMetalBufferPiece<float>& in,const MetalSparseMidpoint& midpoint);
    std::tuple<unsigned int,float> operator()(TypedMetalBufferPiece<float>&,MetalSparseMidpoint&,MetalBufferPiece&);
    void create_threadgroup_memory(MTL::ComputeCommandEncoder* encoder) const override;
};


class MidpointOperationKernel:public KernelFxn{
public:
    MidpointOperationKernel(unsigned int per_thread_compute,MetalCommandQueue *mCommandQueu):KernelFxn(xSTRINGIFY(MDPNT_OPR_KRNL_NM),0,per_thread_compute, mCommandQueu){};
    void encode_arguments(MTL::ComputeCommandEncoder*,
                          TypedMetalBufferPiece<float>& input,
                          const MetalSparseMidpoint&,
                          const float&,
                          const unsigned int&,
                          const unsigned int&);
    void cpu_operation(
                    TypedMetalBufferPiece<float>& ,const MetalSparseMidpoint& ,
                    std::vector<float>&,const unsigned int& ,
                    const float& ,const unsigned int& );
    void operator()(TypedMetalBufferPiece<float>&,
                    const MetalSparseMidpoint& ,
                    const unsigned int&,
                    const float&,
                    const unsigned int&);
};




#endif /* KernelLibrary_hpp */
#endif
