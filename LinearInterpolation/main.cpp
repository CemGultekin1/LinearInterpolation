//
//  main.cpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 3/31/24.
//

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <iostream>
#include "MetalSupport/KernelLibrary.hpp"



int main(){
    MTL::Device *device = MTL::CreateSystemDefaultDevice();
    auto mQueue = new MetalCommandQueue(device);
    auto gMem = new GlobalMetalMemory(device);
    
    unsigned int nnz_dim = 999'999;
    unsigned int origin = nnz_dim + 1;
    unsigned int per_thread = 1024;
    
    FirstLayerKernel flk(per_thread, mQueue,true);
    unsigned int padded_dim = flk.configure(nnz_dim);
    std::cout << "padded_dim = " << padded_dim <<", nnz_dim = " << nnz_dim << "\n";
    SizedMetalBuffer<float> input{padded_dim,origin};
    gMem->reserve_gpu_memory(input);
    gMem->fill_with_random(input, nnz_dim);
    
    SizedMetalBuffer<float> input1{padded_dim,origin};
    gMem->reserve_gpu_memory(input1);
    gMem->fill_with_random(input1, nnz_dim);
    
    
    SizedMetalBuffer<float> temp = flk.memory_request();
    gMem->reserve_gpu_memory(temp);
    flk(input,temp);
    
    gMem->add_midpoint(input,1e-7);
    auto midpoint = gMem->_midpoints.back();
    std::cout << midpoint.to_string() << "\n";
    
    
    
    flk(input1,temp);
    
    FindExitKernel fek(per_thread,mQueue,true);
    fek.configure(midpoint.weights.size);
    std::cout << "midpoint.weights.size : " << midpoint.weights.size << "\n";
    std::cout << "fek._num_threadgroups : " << fek._num_threadgroups << "\n";
    std::cout << "fek._threadgroup_size : " << fek._threadgroup_size << "\n";
    std::cout << "#numthreads           : " << fek._num_threadgroups*fek._threadgroup_size << "\n";
    auto [x,y,z] = fek.memory_request();
    gMem->reserve_gpu_memory(x);
    gMem->reserve_gpu_memory(y);
    gMem->reserve_gpu_memory(z);
    auto [success_flag,exit_index,time_step] = fek(input1, midpoint, x, y, z);
    std::cout << "success_flag = "<< success_flag << "\n";
    std::cout << "exit_index = " << exit_index << "\n";
    
    
    MidpointOperationKernel mok{per_thread,mQueue,true};
    mok.configure(midpoint.weights.size);
    mok(input1,midpoint,time_step,exit_index,0);
    gMem->add_midpoint(input1,1e-7);
    
    
    SizedMetalBuffer<float> input2{padded_dim,origin};
    gMem->reserve_gpu_memory(input2);
    gMem->fill_with_random(input2, nnz_dim);
    
    auto midpoint2 = gMem->_midpoints.back();
    
    flk(input2,temp);
    auto [success_flag2,exit_index2,time_step2] = fek(input2, midpoint, x, y, z);
    std::cout << "exit_index2 = " <<exit_index2 << "\n";
}

//#include "simplex_tree.hpp"
//#include "debug_display.hpp"
//#include "random_generators.hpp"
//#include "facing_node.hpp"
//#include "numerical_debug.hpp"
//int main(){
//    int d = int(1e2);
//    SimplexTree st(0, 1e-9);
//    st.midpoint_table.max_dim = d;
//    InterpolationTest intest{&st};
//    srand(1);
//    for(int i = 0; i <16; i ++ ){
//        std::vector<float> x{};
//        random_float_vector(&x, d);
//        auto alias = st.add_midpoint(x);
//        if(alias == std::numeric_limits<size_t>::max()){
//            continue;
//        }
//        std::cout << "#node = " << st.midpoint_table.data.size() <<", ";
//        std::cout << "#midpoint = " << st.midpoint_table.midpoints.size() << ", ";
//        std::cout << " depth = " <<
//                    st.midpoint_table.depth << "\n";
//        log_sink.report();
//        std::cout << " --------------- " << "\n";
//    }
//    for(int i = 0; i < 12; i++){
//        std::vector<float> x{};
//        random_float_vector(&x, d);
//        auto err = intest(x);
//        std::cout << "test # " << i << " err = ";
//        sciprint(err,2);
//        std::cout << "\n";
//    }
//}

