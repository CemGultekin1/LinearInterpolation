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

static float r2_value(float*x,float*y,size_t n){
    float err = 0 ;
    float norm = 0;
    for(unsigned int i = 0; i < n ; i++,x++,y++ ){
        err += std::pow(*x - *y, 2);
        norm += (std::pow(*x,2) + std::pow(*y,2))/2;
    }
    return 1 - err/norm;
}


void KernelConfiguration::set_dim(unsigned int _dim){
    _dim += 1;// one more spot is needed for d+1th dimension
    auto num_threads = ceil_up_to_divisible(_dim,grain_size);
    auto max_per_threadgroup = device->maxThreadsPerThreadgroup().width;
    if(per_thread_threadgroup_bytes != 0){
        max_per_threadgroup = std::min(max_per_threadgroup,device->maxThreadgroupMemoryLength()/per_thread_threadgroup_bytes);
    }
    if(num_threads > max_per_threadgroup){
        num_threadgroups = (num_threads-1)/max_per_threadgroup+1;
    }else{
        num_threadgroups = 1;
    }
    threadgroup_size = ceil_up_to_divisible((num_threads - 1)/num_threadgroups + 1,warpsize)*warpsize;
    grid_size = threadgroup_size*num_threadgroups;
    dim = _dim;
    padded_dim = grid_size*grain_size;
    temporary_bytes = num_threads*per_thread_threadgroup_bytes;
    threadgroup_bytes = per_thread_threadgroup_bytes*threadgroup_size;
    
    fpdim = static_cast<float>(padded_dim - 1);
}


unsigned int KernelConfiguration::rounded_up_dimension(unsigned int new_dim){
    auto dim0 = dim;
    set_dim(new_dim);
    auto _padded_dim = padded_dim;
    set_dim(dim0);
    return _padded_dim;
}

KernelConfiguration::KernelConfiguration(MTL::Device* device,
                                         unsigned int _dim,
                                         unsigned int _grain_size, 
                                         unsigned int _per_thread_threadgroup_bytes,
                                         unsigned int warpsize){
    grain_size = _grain_size;
    this->device = device;
    this->warpsize = warpsize;
    per_thread_threadgroup_bytes = _per_thread_threadgroup_bytes;
    set_dim(_dim);
    
}


std::string KernelConfiguration::to_string() const{
    std::string x = "";
    x +=  
        std::to_string(dim) + " <= " +
        std::to_string(padded_dim) + " = " +
        std::to_string(grain_size)  + " x " +
        std::to_string(threadgroup_size)  + " x " +
        std::to_string(num_threadgroups)  + " : " +
        std::to_string(temporary_bytes) + "B";
    return x;
}

struct ArgumentSetter{
    int arg_counter;
    MTL::ComputeCommandEncoder* encoder;
    ArgumentSetter(MTL::ComputeCommandEncoder* _encoder){
        arg_counter = 0;
        encoder = _encoder;
    };
    void setArgument(const MetalBufferPiece& mbc){
        encoder->setBuffer(mbc.get_buffer(), mbc.get_byte_offset(), arg_counter++);
    }


    void setArgument(const MetalSparseMidpoint& msm){
        encoder->setBuffer(msm.nodes.get_buffer(), msm.nodes.get_byte_offset(), arg_counter++);
        encoder->setBuffer(msm.weights.get_buffer(), msm.weights.get_byte_offset(), arg_counter++);
        setArgument(msm.numel());
    }

    void setArgument(unsigned int x){
        encoder->setBytes(&x, sizeof(x), arg_counter++);
    }

    void setArgument(float x){
        encoder->setBytes(&x, sizeof(x), arg_counter++);
    }
};


KernelFxn::KernelFxn(std::string  kernel_root,
                     unsigned int per_thread_threadgroup_memory,
                     unsigned int per_thread_compute,
                     MetalCommandQueue *mCommandQueue){
    _kernel_root = kernel_root;
    _mcq = mCommandQueue;
    _unique_kernel_name = _kernel_root + "_pt_" + std::to_string(per_thread_compute);
    NS::Error *error = nullptr;
    _mDevice = mCommandQueue->_mDevice;
    auto fxn_str = NS::String::string(_unique_kernel_name.c_str(), NS::ASCIIStringEncoding);
    MTL::Library *defaultLibrary = _mDevice->newDefaultLibrary();
    MTL::Function *fxn = defaultLibrary->newFunction(fxn_str);
    _compute_pipeline_state_ptr = _mDevice->newComputePipelineState(fxn, &error);
    conf = new KernelConfiguration(_mDevice, 0, per_thread_compute,per_thread_threadgroup_memory);
}

std::pair<MTL::CommandBuffer*, MTL::ComputeCommandEncoder*> KernelFxn::create_encoder_buffer() const{
    auto buffer = _mcq->create_buffer();
    auto encoder = buffer->computeCommandEncoder();
    assert(encoder != nullptr);
    encoder->setComputePipelineState(_compute_pipeline_state_ptr);    
    return {buffer,encoder};
}

void KernelFxn::create_threadgroup_memory(MTL::ComputeCommandEncoder* encoder) const{
    if(conf->threadgroup_bytes != 0){
        encoder->setThreadgroupMemoryLength(conf->threadgroup_bytes, 0);
    }
}

unsigned int KernelFxn::needed_temporary_bytes(unsigned int dim) const{
    auto d0 = conf->dim;
    conf->set_dim(dim);
    auto tb =  conf->temporary_bytes;
    conf->set_dim(d0);
    return tb;
}


void KernelFxn::free_encoder_buffer(MTL::CommandBuffer* buffer, MTL::ComputeCommandEncoder* encoder) const{
    create_threadgroup_memory(encoder);
    MTL::Size threadgroupSize = MTL::Size::Make(conf->threadgroup_size, 1, 1);
    MTL::Size gridSize = MTL::Size::Make(conf->grid_size, 1, 1);
    encoder->dispatchThreads(gridSize, threadgroupSize);
    encoder->endEncoding();
    buffer->commit();
    buffer->waitUntilCompleted();
}

//------------------------------------------------------------------------------------------------------------//
//------------------------------------------------------------------------------------------------------------//
//--------------------------------------------|FirstLayerKernel|----------------------------------------------//
//------------------------------------------------------------------------------------------------------------//
//------------------------------------------------------------------------------------------------------------//


void FirstLayerKernel::encode_arguments(MTL::ComputeCommandEncoder* encoder,TypedMetalBufferPiece<float>& input, MetalBufferPiece& summem) const{
    ArgumentSetter args{encoder};
    args.setArgument(input);
    args.setArgument(summem);
    args.setArgument(min);
    args.setArgument(max);
    args.setArgument(mean);
    args.setArgument(conf->fpdim);
}



float FirstLayerKernel::cpu_operation(TypedMetalBufferPiece<float>& input, float *out) const{
    float* in = input.content_pointer();
    float _mean_hat = 0;
    float dim = conf->fpdim;
    for(unsigned int i = 0; i < conf->padded_dim; i ++){
        out[i] = in[i]/dim;
        assert(out[i] >= 0);
        _mean_hat += out[i];
    }
    if(single_threadblock()){        
        float kappa = kappa_fun(mean, dim);
        float beta = beta_fun(_mean_hat, dim, kappa);
        for(unsigned int i = 0; i < conf->padded_dim; i ++){
            out[i] += beta;
        }
        out[conf->padded_dim - 1] = last_entry_fun(_mean_hat, kappa);
    }
    return _mean_hat;
}

float FirstLayerKernel::operator()(TypedMetalBufferPiece<float>& input, MetalBufferPiece& temp) const{
    auto [buffer,encoder] = create_encoder_buffer();
    
    
    
#ifdef NUMERICAL_CPU_CHECK
    std::vector<float> out;
    float _cpu_mean_hat;
    out.resize(conf->padded_dim);
    _cpu_mean_hat = cpu_operation(input, &out[0]);
#endif
    
    encode_arguments(encoder, input, temp);
        
    free_encoder_buffer(buffer, encoder);
    
    auto partial_sums = temp.content_pointer<float>();
    float _mean_hat = 0;
    for(int i = 0; i < conf->num_threadgroups; i++ ){
        _mean_hat += partial_sums[i];
    }
#ifdef NUMERICAL_CPU_CHECK
        float r2_vector = r2_value(input.content_pointer(), &out[0], out.size());
        float r2_mean = r2_value(&_mean_hat, &_cpu_mean_hat, 1);
        DEBUG_PRINT("%s : r2-vector = %f, r2-mean = %f\n",_unique_kernel_name.c_str(),r2_vector,r2_mean);
        
#endif
    return _mean_hat;
}

void PosteriorFirstLayerKernel::encode_arguments(MTL::ComputeCommandEncoder *encoder, TypedMetalBufferPiece<float> &input, float &_mean) const{
    ArgumentSetter args{encoder};
    args.setArgument(input);
    args.setArgument(_mean);
    args.setArgument(conf->fpdim);
}
void PosteriorFirstLayerKernel::cpu_operation(TypedMetalBufferPiece<float> &input, float &_mean, float *out) const{
    float* in = input.content_pointer();
    float dim = conf->fpdim;
    float kappa = kappa_fun(_mean,  dim);//dim*(1 - _mean*(dim+1)/dim);
    float beta =  beta_fun(_mean, dim, kappa);//(1-_mean)*kappa/dim/(kappa + 1);
    for(unsigned int i = 0; i < conf->padded_dim; i ++){
        out[i] = in[i] + beta;
        assert(out[i] >= 0);
    }
    out[conf->padded_dim - 1] = last_entry_fun(_mean, kappa);//(1-_mean)/(1+kappa);
}


void PosteriorFirstLayerKernel::operator()(TypedMetalBufferPiece<float>& input, float & mean) const{
    auto [buffer,encoder] = create_encoder_buffer();
    
#ifdef NUMERICAL_CPU_CHECK
        std::vector<float> out;
        out.resize(conf->padded_dim);
        cpu_operation(input, mean,&out[0]);
#endif
    encode_arguments(encoder, input, mean);
    free_encoder_buffer(buffer, encoder);
#ifdef NUMERICAL_CPU_CHECK
        float r2 = r2_value(input.content_pointer(), &out[0], out.size());
        DEBUG_PRINT("%s : r2 = %f \n",_unique_kernel_name.c_str(),r2);
#endif
}



//------------------------------------------------------------------------------------------------------------//
//------------------------------------------------------------------------------------------------------------//
//--------------------------------------------| FindExitKernel |----------------------------------------------//
//------------------------------------------------------------------------------------------------------------//
//------------------------------------------------------------------------------------------------------------//

  
void FindExitKernel::encode_arguments(MTL::ComputeCommandEncoder* encoder,
                                      TypedMetalBufferPiece<float>& input,
                                      MetalSparseMidpoint& midpoint,
                                      MetalBufferPiece& glbl_min_indexes,
                                      MetalBufferPiece& glbl_min_weights){
    ArgumentSetter args{encoder};
    args.setArgument(input);
    args.setArgument(midpoint);
    
    args.setArgument(glbl_min_indexes);
    args.setArgument(glbl_min_weights);
}

void FindExitKernel::create_threadgroup_memory(MTL::ComputeCommandEncoder* encoder) const{
    if(conf->per_thread_threadgroup_bytes == 0){
        return;
    }
    encoder->setThreadgroupMemoryLength(sizeof(unsigned int)*conf->threadgroup_size, 0);
    encoder->setThreadgroupMemoryLength(sizeof(float)*conf->threadgroup_size, 1);
    encoder->setThreadgroupMemoryLength(sizeof(char)*conf->threadgroup_size, 2);
}

std::tuple<unsigned int,float> FindExitKernel::cpu_operation(TypedMetalBufferPiece<float>& input,const MetalSparseMidpoint& midpoint){
    auto x = input.content_pointer();
    
    unsigned int exit_index = 0;
    unsigned int n = midpoint.numel();
    float min_weight = MAXFLOAT;
    
    auto nodes = midpoint.nodes.content_pointer();
    auto weights = midpoint.weights.content_pointer();
    for(unsigned int i = 0 ; i < n; i ++ ){
        auto j = nodes[i];
        auto mw = weights[i];
        auto xw = x[j];
        mw = division_zero_permissible(xw, mw);
        if(mw < min_weight){
            min_weight = mw;
            exit_index = i;
        }
    }
    return {exit_index,min_weight};
    
}
std::tuple<unsigned int,float> FindExitKernel::operator()(
                                TypedMetalBufferPiece<float>& input,
                                MetalSparseMidpoint& midpoint,
                                MetalBufferPiece& temp){
    conf->set_dim(midpoint.numel());
    DEBUG_PRINT("%s \n", to_string().c_str());
    auto [buffer,encoder] = create_encoder_buffer();
    
    
    MetalBufferSlicer slicer{temp};
    
    auto glbl_min_indexes = slicer.step_forward<unsigned int>(conf->num_threadgroups);
    auto glbl_min_weights = slicer.step_forward<float>(conf->num_threadgroups);
    
    
    encode_arguments(encoder, input, midpoint,glbl_min_indexes,glbl_min_weights);
    free_encoder_buffer(buffer, encoder);
    
    auto indexes_ptr = glbl_min_indexes.content_pointer<unsigned int>();
    auto weights_ptr = glbl_min_weights.content_pointer<float>();
    
    unsigned int global_index = 0;
    float gpu_step_size = MAXFLOAT;
    
    for(int i = 0; i < conf->num_threadgroups; i++ ){
        auto index = indexes_ptr[i];
        auto weight = weights_ptr[i];
    
        if(gpu_step_size > weight){
            gpu_step_size = weight;
            global_index = index;
        }
    }
#ifdef NUMERICAL_CPU_CHECK
        auto [cpu_global_index,cpu_step_size] = cpu_operation(input, midpoint);
        DEBUG_PRINT("\t\t gpu exit_index = %d\n\t\t\t cpu exit_index = %d \n"
                    "\t\t\t gpu time_step = %s\n\t\t\t cpu time_step = %s \n"
                    "\t\t\t gpu x/m = %s/%s\n\t\t\t cpu x/m = %s/%s \n",
                    global_index,cpu_global_index,
                    SCI_FL_STR(gpu_step_size),
                    SCI_FL_STR(cpu_step_size),
                    SCI_FL_STR(input.content_pointer()[midpoint.nodes.content_pointer()[global_index]]),
                    SCI_FL_STR(midpoint.weights.content_pointer()[global_index]),
                    SCI_FL_STR(input.content_pointer()[midpoint.nodes.content_pointer()[cpu_global_index]]),
                    SCI_FL_STR(midpoint.weights.content_pointer()[cpu_global_index]));
    
#endif
    return {global_index,gpu_step_size};
}




//------------------------------------------------------------------------------------------------------------//
//------------------------------------------------------------------------------------------------------------//
//-----------------------------------------|MidpointOperationKernel|------------------------------------------//
//------------------------------------------------------------------------------------------------------------//
//------------------------------------------------------------------------------------------------------------//


void MidpointOperationKernel::cpu_operation(
TypedMetalBufferPiece<float>& input,const MetalSparseMidpoint& midpoint,
std::vector<float>&out,const unsigned int& escape_index,
const float& time_step,const unsigned int& midpoint_node){
    auto midpoint_nodes = midpoint.nodes.content_pointer();
    auto midpoint_values = midpoint.weights.content_pointer();
    for(unsigned int i = 0 ; i < midpoint.nodes.numel(); i ++){
        auto index = midpoint_nodes[i];
        out[index] = midpoint_values[i]*(1 - time_step);
    }
    auto escape_node = midpoint_nodes[escape_index];
    out[escape_node] = 0;
    out[midpoint_node] = time_step;
}


void MidpointOperationKernel::operator()(TypedMetalBufferPiece<float>& input,const MetalSparseMidpoint& midpoint,
     const unsigned int& escape_index,const float& time_step,const unsigned int& midpoint_node){
    conf->set_dim(midpoint.numel());
    DEBUG_PRINT("%s \n", to_string().c_str());
    auto [buffer,encoder] = create_encoder_buffer();
    
    std::vector<float> out;
#ifdef NUMERICAL_CPU_CHECK
        out.resize(input.numel(),0);
        cpu_operation(input, midpoint,out,escape_index,time_step,midpoint_node);
#endif
    
    encode_arguments(encoder, input, midpoint,time_step,escape_index,midpoint_node);
    free_encoder_buffer(buffer, encoder);
    
#ifdef NUMERICAL_CPU_CHECK
        float r2 = r2_value(input.content_pointer(), &out[0], out.size());
        DEBUG_PRINT("\t\t r2 = %f \n", r2);
#endif
}
void MidpointOperationKernel::encode_arguments(
           MTL::ComputeCommandEncoder* encoder,
           TypedMetalBufferPiece<float>& input,
           const MetalSparseMidpoint& midpoint,
           const float& time_step,
           const unsigned int& escape_index,
           const unsigned int& midpoint_node){
    ArgumentSetter args{encoder};
    args.setArgument(input);
    args.setArgument(midpoint);
    args.setArgument(time_step);
    args.setArgument(escape_index);
    args.setArgument(midpoint_node);
}


