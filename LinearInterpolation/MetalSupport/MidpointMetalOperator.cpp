////
////  MetalAdder.cpp
////  LinearInterpolation
////
////  Created by Cem Gultekin on 5/10/24.
////
//
//#include "MidpointMetalOperator.hpp"
//#include <iostream>
//#include <bit>
//#include <algorithm>
//#include <exception>
//
//unsigned int ceil_up_to_divisible(unsigned int x, unsigned int y){
//    return 1 + ((x - 1) / y);
//}
//
//
//MetalOperation::MetalOperation(MTL::Device *device)
//{
//
//    _mDevice = device;
//
//    
//
//    _mCommandQueue = _mDevice->newCommandQueue();
//    if (_mCommandQueue == nullptr)
//    {
//        std::cout << "Failed to find the command queue." << std::endl;
//        return;
//    }
//}
//MetalOperation::~MetalOperation(){
//    if(_pso != nullptr){
//        _pso->release();
//    }
//    if(_mCommandQueue != nullptr){
//        _mCommandQueue->release();
//    }
//}
//
//void MetalOperation::setup_pso(){
//    if(_pso != nullptr){
//        _pso->release();
//    }
////     Load the shader files with a .metal file extension in the project
//    MTL::Library *defaultLibrary = _mDevice->newDefaultLibrary();
//    
//    if (defaultLibrary == nullptr)
//    {
//        std::cout << "Failed to find the default library." << std::endl;
//        return;
//    }
//    auto str = NS::String::string(get_fxn_name().c_str(), NS::ASCIIStringEncoding);
//    MTL::Function *fxn = defaultLibrary->newFunction(str);
//    defaultLibrary->release();
//    
//    if (fxn == nullptr)
//    {
//        std::cout << "Failed to find the function "<< get_fxn_name().c_str() << std::endl;
//        return;
//    }
//    NS::Error *error = nullptr;
//    // Create a compute pipeline state object.
//    _pso = _mDevice->newComputePipelineState(fxn, &error);
//    fxn->release();
//    
//    if (_pso == nullptr)
//    {
//        std::cout << "Failed to created pipeline state object, error " << error << "." << std::endl;
//        return;
//    }
//}
//
//FirstLayerOperator::FirstLayerOperator(MTL::Device *device,unsigned int per_thread_compute):MetalOperation(device){
//    _per_thread_compute = per_thread_compute;
//}
//unsigned int MetalOperation::configure(unsigned int input_size){
//    auto num_threads = ceil_up_to_divisible(input_size,_per_thread_compute);
//    auto max_per_threadgroup = std::min(_mDevice->maxThreadsPerThreadgroup().width,
//                                        _mDevice->maxThreadgroupMemoryLength()/_per_threadgroup_memory);
//    if(num_threads > max_per_threadgroup){
//        _num_threadgroups = (num_threads-1)/max_per_threadgroup+1;
//    }else{
//        _num_threadgroups = 1;
//    }
//    _threadgroup_size = (num_threads - 1)/_num_threadgroups + 1;
//    _threadgroup_size = ceil_up_to_divisible(_threadgroup_size,32)*32;
//    _grid_size = _threadgroup_size*_num_threadgroups;
//    return _threadgroup_size*_num_threadgroups*_per_thread_compute;
//}
//std::string MetalOperation::get_fxn_name(){
//    return kernel_name + "_per_thread_" + std::to_string(_per_thread_compute);
//}
//
//void MetalOperation::send_compute_command(MetalMemory* mm)
//{
//    // Create a command buffer to hold commands.
//    MTL::CommandBuffer *commandBuffer = _mCommandQueue->commandBuffer();
//    assert(commandBuffer != nullptr);
//
//    // Start a compute pass.
//    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
//    assert(computeEncoder != nullptr);
//
//    encode_add_command(mm,computeEncoder);
//
//    // End the compute pass.
//    computeEncoder->endEncoding();
//
//    // Execute the command.
//    commandBuffer->commit();
//
//    // Normally, you want to do other work in your app while the GPU is running,
//    // but in this example, the code simply blocks until the calculation is complete.
//    commandBuffer->waitUntilCompleted();
//    auto partial_sums = (float*) mm->_temporary.buffer->contents();
//    float _sum_of_all = 0;
//    for(int i = 0; i < _num_threadgroups; i++ ){
//        _sum_of_all += partial_sums[i];
//    }
//    auto lastel = mm->_input.unpadded_num_elems +(float*) mm->_input.buffer->contents();
////    assert(_sum_of_all <= 1.);
//    *lastel = 1.- _sum_of_all;
//    mm->_input.unpadded_num_elems++;
//    
//}
//
//void MetalOperation::encode_add_command(MetalMemory* mm,MTL::ComputeCommandEncoder *computeEncoder)
//{
//    // Encode the pipeline state object and its parameters.
//    
//    computeEncoder->setComputePipelineState(_pso);
//    MTL::Size gridSize = MTL::Size::Make(_grid_size, 1, 1);
//    
//
//    // Calculate a threadgroup size.MTL::ResourceStorageModeShared);
////    NS::UInteger threadGroupSize = _firstLayerPSO->maxTotalThreadsPerThreadgroup();
////    if (threadGroupSize > _input.size/num_thread_groups)
////    {
////        threadGroupSize = _input.size/num_thread_groups;
////    }
//    
//
//    setup_inputs(mm, computeEncoder);
//    
//    unsigned int nbytesThreadgroup = _threadgroup_size*sizeof(float);
//    unsigned int minimum_threadgroup_byte = 16;
//    nbytesThreadgroup = std::max(nbytesThreadgroup, minimum_threadgroup_byte);
//    computeEncoder->setThreadgroupMemoryLength(nbytesThreadgroup, 0);
//    // Encode the compute command.
//    MTL::Size threadgroupSize = MTL::Size::Make(_threadgroup_size, 1, 1);
//    computeEncoder->dispatchThreads(gridSize, threadgroupSize);
//}
//
//void FirstLayerOperator::setup_inputs(MetalMemory* mm,MTL::ComputeCommandEncoder *computeEncoder){
//    mm->request_temporary_space(_num_threadgroups,sizeof(float));
//    auto _input = mm->_input;
//    computeEncoder->setBuffer(_input.buffer, 0, 0);
//    computeEncoder->setBuffer(mm->_temporary.buffer,0, 1);
//    computeEncoder->setBytes(&_input.elem_size, sizeof(_input.elem_size), 2);
//}
//
//
//bool FirstLayerOperator::verify_computation(MetalMemory* mm ){
//    auto ds = (float*)mm->_input.buffer->contents();
//    float z = 0;
//    bool flag = true;
//    for(int i = 0 ; i < mm->_input.unpadded_num_elems; i ++ ){
//        z += ds[i];
//        flag &= (ds[i] >= 0);
//        if(ds[i] < 0){
//            throw std::invalid_argument("negative weight found "  +  std::to_string(ds[i]) +  "\n");
//        }
//    }
//    float error_lim = 1e-3;
//    flag &= std::fabs(z - 1) < error_lim;
//    if(std::fabs(z - 1) > error_lim){
////        throw std::invalid_argument("sum of weights "  +  std::to_string(z) +  "\n");
//        std::cout <<"sum of weights "  +  std::to_string(z) +  "\n";
//    }
//    return flag;
//}
//
//MetalMemory::MetalMemory(MTL::Device * device,unsigned int grain_size){
//    _grain_size = grain_size;
//    _mDevice = device;
//}
//
//unsigned int MetalMemory::internal_input_dimension(unsigned int dimension) const{
//    return static_cast<unsigned int>(_midpointNodes.size())+dimension + 1;
//}
//
//void MetalMemory::generate_random_float_input(unsigned int dimension,unsigned int zero_filled)
//{
//    _input.elem_size = zero_filled;
//    _input.byte_size = _input.elem_size*sizeof(float);
//    _input.unpadded_num_elems = dimension;
//    _input.buffer = _mDevice->newBuffer(_input.byte_size, MTL::ResourceStorageModeShared);
//    float *dataPtr = (float *)_input.buffer->contents();
//
//    for (unsigned int index = 0; index < dimension; index++)
//    {
//        dataPtr[index] = (float) rand() / (float)(RAND_MAX);
//    }
//    for(unsigned int index = dimension; index < zero_filled; index ++){
//        dataPtr[index] = 0;
//    }
//}
//
//void MetalMemory::print_input(){
//    float *dataPtr = (float *)_input.buffer->contents();
//    for (unsigned int index = 0; index < _input.elem_size; index++)
//    {
//        std::cout << index <<" : "<< dataPtr[index] << "\n";
//    }
//}
//
//void MetalMemory::request_temporary_space(unsigned int elem_size,unsigned int sizeof_elem){
//    if(_temporary.byte_size < elem_size*sizeof_elem){
//        if(_temporary.buffer != nullptr){
//            _temporary.buffer->release();
//        }
//        _temporary.buffer = _mDevice->newBuffer(elem_size*sizeof_elem, MTL::ResourceStorageModeShared);
//        _temporary.elem_size = elem_size;
//        _temporary.byte_size = elem_size*sizeof_elem;
//    }
//}
//MetalMemory::~MetalMemory()
//{
//    _input.buffer->release();
//    for(auto &buffVec: {_midpointNodes,_midpointWeights}){
//        for(auto &x: buffVec){
//            x.buffer->release();
//        }
//    }
//}
//
////
////
////unsigned int MetalMemory::to_midpoint(){
////    auto x = (float*) _input.buffer->contents();
////    
////}
