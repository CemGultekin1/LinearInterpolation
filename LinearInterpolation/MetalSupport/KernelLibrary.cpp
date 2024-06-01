//
//  KernelLibrary.cpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 5/19/24.
//

#include "KernelLibrary.hpp"

static unsigned int ceil_up_to_divisible(unsigned int x, unsigned int y){
    return 1 + ((x - 1) / y);
}

static float l2_rel_err(float*x,float*y,size_t n){
    float err = 0 ;
    float norm = 0;
    for(unsigned int i = 0; i < n ; i++,x++,y++ ){
        err += std::pow(*x - *y, 2);
        norm += std::pow(*x,2);
    }
    return err/norm;
}


KernelFxn::KernelFxn(std::string  kernel_root,unsigned int per_thread_threadgroup_memory,unsigned int per_thread_compute,MetalCommandQueue *mCommandQueue,bool cpu_verify){
    _kernel_root = kernel_root;
    _per_thread_threadgroup_memory = per_thread_threadgroup_memory;
    _mcq = mCommandQueue;
    _unique_kernel_name = _kernel_root + "_pt_" + std::to_string(per_thread_compute);
    NS::Error *error = nullptr;
    _mDevice = mCommandQueue->_mDevice;
    auto str = NS::String::string(_unique_kernel_name.c_str(), NS::ASCIIStringEncoding);
    MTL::Library *defaultLibrary = _mDevice->newDefaultLibrary();
    MTL::Function *fxn = defaultLibrary->newFunction(str);
    _compute_pipeline_state_ptr = _mDevice->newComputePipelineState(fxn, &error);
    _thread_compute = per_thread_compute;
    _cpu_verify_flag = cpu_verify;
}
unsigned int  KernelFxn::configure(unsigned int input_size){
    input_size += 1;// one more spot is needed for d+1th dimension
    auto num_threads = ceil_up_to_divisible(input_size,_thread_compute);
    auto max_per_threadgroup = _mDevice->maxThreadsPerThreadgroup().width;
    if(_per_thread_threadgroup_memory != 0){
        max_per_threadgroup = std::min(max_per_threadgroup,_mDevice->maxThreadgroupMemoryLength()/_per_thread_threadgroup_memory);
    }
    if(num_threads > max_per_threadgroup){
        _num_threadgroups = (num_threads-1)/max_per_threadgroup+1;
    }else{
        _num_threadgroups = 1;
    }
    _threadgroup_size = ceil_up_to_divisible((num_threads - 1)/_num_threadgroups + 1,32)*32;
    _grid_size = _threadgroup_size*_num_threadgroups;
    return _threadgroup_size*_num_threadgroups*_thread_compute;
}
std::pair<MTL::CommandBuffer*, MTL::ComputeCommandEncoder*> KernelFxn::create_encoder_buffer(){
    auto buffer = _mcq->create_buffer();
    auto encoder = buffer->computeCommandEncoder();
    assert(encoder != nullptr);
    encoder->setComputePipelineState(_compute_pipeline_state_ptr);    
    return {buffer,encoder};
}

void KernelFxn::create_threadgroup_memory(MTL::ComputeCommandEncoder* encoder){
    if(_per_thread_threadgroup_memory != 0){
        encoder->setThreadgroupMemoryLength(_per_thread_threadgroup_memory*_threadgroup_size, 0);
    }
}
void KernelFxn::free_encoder_buffer(MTL::CommandBuffer* buffer, MTL::ComputeCommandEncoder* encoder){
    create_threadgroup_memory(encoder);
    MTL::Size threadgroupSize = MTL::Size::Make(_threadgroup_size, 1, 1);
    MTL::Size gridSize = MTL::Size::Make(_grid_size, 1, 1);
    encoder->dispatchThreads(gridSize, threadgroupSize);
    encoder->endEncoding();
    buffer->commit();
    buffer->waitUntilCompleted();
}


void  FirstLayerKernel::encode_arguments(MTL::ComputeCommandEncoder* encoder,SizedMetalBuffer<float>& input, SizedMetalBuffer<float>& temp){
//    _gmm->request_temporary_space(_num_threadgroups*_per_thread_threadgroup_memory,sizeof(float));
    encoder->setBuffer(input.buffer, 0, 0);
    encoder->setBuffer(temp.buffer,0, 1);
}


SizedMetalBuffer<float> FirstLayerKernel::memory_request(){
    SizedMetalBuffer<float> x{_num_threadgroups};
    return x;
}

void FirstLayerKernel::cpu_operation(SizedMetalBuffer<float>& input, float *out){
    float* in = (float*) input.buffer->contents();
    float _sum = 0;
    unsigned int n    = _thread_compute * _num_threadgroups * _threadgroup_size;
    float sc = n - 1;
    for(unsigned int i = 0; i < n ; i ++){
        out[i] = in[i]/sc;
        assert(out[i] >= 0);
        _sum += out[i];
    }
    out[input.origin-1] = 1. - _sum;
}

bool FirstLayerKernel::operator()(SizedMetalBuffer<float>& input, SizedMetalBuffer<float>& temp){
    auto [buffer,encoder] = create_encoder_buffer();
    
    
    std::vector<float> out;
    if(_cpu_verify_flag){
        out.resize(_thread_compute * _num_threadgroups * _threadgroup_size);
        cpu_operation(input, &out[0]);
    }
    
    encode_arguments(encoder, input, temp);
    
    
    free_encoder_buffer(buffer, encoder);
    
    auto partial_sums = (float*) temp.buffer->contents();
    float _sum_of_all = 0;
    for(int i = 0; i < _num_threadgroups; i++ ){
        _sum_of_all += partial_sums[i];
    }
    auto vals = (float*) input.buffer->contents();
    vals[input.origin-1] = 1.- _sum_of_all;
    if(_cpu_verify_flag){
        float err = l2_rel_err((float*) input.buffer->contents(), &out[0], out.size());
        std::cout << "err = " << err << "\n";
    }
    return true;
}

std::tuple<SizedMetalBuffer<unsigned int>,SizedMetalBuffer<float>,SizedMetalBuffer<char>>
FindExitKernel::memory_request(){
    unsigned int n = _num_threadgroups;
    SizedMetalBuffer<unsigned int> x{n};
    SizedMetalBuffer<float> y{n};
    SizedMetalBuffer<char> z{n};
    return {x,y,z};
}
    
void FindExitKernel::encode_arguments(MTL::ComputeCommandEncoder* encoder,
                                      SizedMetalBuffer<float>& input,
                                      Midpoint& midpoint,
                                      SizedMetalBuffer<unsigned int>& glbl_min_indexes,
                                      SizedMetalBuffer<float>& glbl_min_weights,
                                      SizedMetalBuffer<char>& glbl_legal){
    encoder->setBuffer(input.buffer, 0, 0);
    encoder->setBytes(&input.origin, sizeof(input.origin), 1);
    encoder->setBuffer(midpoint.nodes.buffer, 0, 2);
    encoder->setBuffer(midpoint.weights.buffer, 0, 3);
    encoder->setBytes(&midpoint.nodes.size, sizeof(midpoint.nodes.size), 4);
    encoder->setBuffer(glbl_min_indexes.buffer,0, 5);
    encoder->setBuffer(glbl_min_weights.buffer,0, 6);
    encoder->setBuffer(glbl_legal.buffer,0, 7);
}

void FindExitKernel::create_threadgroup_memory(MTL::ComputeCommandEncoder* encoder){
    if(_per_thread_threadgroup_memory != 0){
        encoder->setThreadgroupMemoryLength(sizeof(unsigned int)*_threadgroup_size, 0);
        encoder->setThreadgroupMemoryLength(sizeof(float)*_threadgroup_size, 1);
        encoder->setThreadgroupMemoryLength(sizeof(char)*_threadgroup_size, 2);
    }
}

std::tuple<bool,unsigned int,float> FindExitKernel::cpu_operation(SizedMetalBuffer<float>& input,unsigned int*midpoint_nodes,float*midpoint_weights,unsigned int n){
    auto x = (float*) input.buffer->contents();
    char legal = 1;
    unsigned int exit_index = 0;
    float min_weight = 1.;
    std::cout << "input.origin = " << input.origin << "\n";
    for(unsigned int i = 0 ; i < n; i ++ ){
        auto j = midpoint_nodes[i] + input.origin;
        auto mw = midpoint_weights[i];
        auto xw = x[j>=0 ? j : 0];
        legal &= ((xw > 0)&&(j >= 0));
        mw = xw/mw;
        if(mw < min_weight){
            min_weight = mw;
            exit_index = i;
        }
    }
    std::cout << "cpu_operation max_value = " << std::to_string(exit_index) << " " << std::to_string(min_weight) << "\n";
    return {legal,exit_index,min_weight};
    
}
std::tuple<bool,unsigned int,float> FindExitKernel::operator()(SizedMetalBuffer<float>& input,
                                Midpoint& midpoint,
                                SizedMetalBuffer<unsigned int>& glbl_min_indexes,
                                SizedMetalBuffer<float>& glbl_min_weights,
                                SizedMetalBuffer<char>& glbl_legals){
    auto [buffer,encoder] = create_encoder_buffer();
    
    
    
    encode_arguments(encoder, input, midpoint,glbl_min_indexes,glbl_min_weights,glbl_legals);
    free_encoder_buffer(buffer, encoder);
    
    auto indexes_ptr = (unsigned int*) glbl_min_indexes.buffer->contents();
    auto weights_ptr = (float*) glbl_min_weights.buffer->contents();
    auto legals_ptr = (char*) glbl_legals.buffer->contents();
   
    unsigned int global_index = 0;
    float gpu_step_size = 1.;
    bool legal_flag = true;
    
    for(int i = 0; i < _num_threadgroups; i++ ){
        auto index = indexes_ptr[i];
        auto weight = weights_ptr[i];
        auto legal = legals_ptr[i];
        legal_flag &= (bool) legal;
        if(gpu_step_size > weight){
            gpu_step_size = weight;
            global_index = index;
        }
    }
    if(_cpu_verify_flag){
        auto [cpu_flag, cpu_global_index,cpu_step_size] =
        cpu_operation(input,
                      (unsigned int *) midpoint.nodes.buffer->contents(),
                      (float*) midpoint.weights.buffer->contents(),
                      midpoint.nodes.size);
        std::cout << "        gpu/cpu legal_flag  = " << legal_flag  << " / " << cpu_flag << "\n";
        std::cout << "        gpu/cpu exit_index  = " << global_index << " / " << cpu_global_index << "\n";
        std::cout << "        gpu/cpu time_step   = " << gpu_step_size << " / " << cpu_step_size << "\n";
    }
    return {legal_flag,global_index,gpu_step_size};
}



void MidpointOperationKernel::cpu_operation(
SizedMetalBuffer<float>& input,const Midpoint& midpoint,
std::vector<float>&out,const unsigned int& escape_index,
const float& time_step,const unsigned int& midpoint_node){
    auto midpoint_nodes = (int*) midpoint.nodes.buffer->contents();
    auto midpoint_values = (float*) midpoint.weights.buffer->contents();
    for(unsigned int i = 0 ; i < midpoint.nodes.size; i ++){
        auto index = input.origin + midpoint_nodes[i];
        out[index] = midpoint_values[i]*(1 - time_step);
    }
    auto escape_node = midpoint_nodes[escape_index];
    out[escape_node + input.origin] = 0;
    out[midpoint_node  + input.origin] = time_step;
}


void MidpointOperationKernel::operator()(SizedMetalBuffer<float>& input,const Midpoint& midpoint,
     const unsigned int& escape_index,const float& time_step,const unsigned int& midpoint_node){
    auto [buffer,encoder] = create_encoder_buffer();
    
    std::vector<float> out;
    if(_cpu_verify_flag){
        out.resize(input.size,0);
        cpu_operation(input, midpoint,out,escape_index,time_step,midpoint_node);
    }
    
    encode_arguments(encoder, input, midpoint,time_step,escape_index,midpoint_node);
    free_encoder_buffer(buffer, encoder);
    
    if(_cpu_verify_flag){
        float err = l2_rel_err((float*) input.buffer->contents(), &out[0], out.size());
        std::cout << "MidpointOperationKernel err = " << err << "\n";
    }
}
void MidpointOperationKernel::encode_arguments(
           MTL::ComputeCommandEncoder* encoder,
           SizedMetalBuffer<float>& input,
           const Midpoint& midpoint,
           const float& time_step,
           const unsigned int& escape_index,
           const unsigned int& midpoint_node){
    encoder->setBuffer(input.buffer, 0, 0);
    encoder->setBytes(&input.origin, sizeof(input.origin), 1);
    encoder->setBuffer(midpoint.nodes.buffer, 0, 2);
    encoder->setBuffer(midpoint.weights.buffer, 0, 3);
    encoder->setBytes(&time_step, sizeof(time_step), 4);
    encoder->setBytes(&escape_index, sizeof(escape_index), 5);
    encoder->setBytes(&midpoint_node, sizeof(midpoint_node), 6);
    encoder->setBytes(&midpoint.weights.size, sizeof(midpoint.weights.size), 7);    
}
