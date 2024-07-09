//
//  main.cpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 3/31/24.
//
//
//






#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "QuartzCore/QuartzCore.hpp"

#include <iostream>
#include "debug_control.h"
//#define NO_METAL

#include "MetalSupport/KernelLibrary.hpp"


int main(){
    MTL::Device *device = MTL::CreateSystemDefaultDevice();
    auto mQueue = new MetalCommandQueue(device);

    unsigned int nnz_dim = 9'999'999;
    unsigned int per_thread = 4;
    
    FirstLayerKernel flk(per_thread, mQueue,nnz_dim,0.5,0,1);
    PosteriorFirstLayerKernel pflk(per_thread*2, mQueue,nnz_dim,0.5,0,1);
    
    
    unsigned int padded_dim = flk.conf->rounded_up_dimension(nnz_dim);
    DEBUG_PRINT("padded_dim = %d, nnz_dim = %d  \n", padded_dim, nnz_dim);

    std::vector<TypedMetalBufferPiece<float>> inputs(3, TypedMetalBufferPiece<float>{padded_dim});
    std::for_each(inputs.begin(),inputs.end(),[device,nnz_dim] (TypedMetalBufferPiece<float>& x) -> void {
        x.to(device);
        fill_with_random(x, nnz_dim, 0);
    });
    
    MetalBufferPiece temp{flk.needed_temporary_bytes(inputs[0].numel()*2),0};
    temp.to(device);
    
    
    
    auto full_first_layer = [&flk,&temp,&pflk](TypedMetalBufferPiece<float>& x){
        float _sum = flk(x,temp);
        if(!flk.single_threadblock()){
            pflk(x,_sum);
        }
    };
    full_first_layer(inputs[0]);
    
    MetalSparseMidpoint midpoint{inputs[0], 1e-5, device, 1};

    full_first_layer(inputs[1]);
    
    FindExitKernel fek(per_thread,mQueue);

    auto [exit_index,time_step] = fek(inputs[1], midpoint, temp);
    
    
    MidpointOperationKernel mok{per_thread,mQueue};
    mok(inputs[1],midpoint,time_step,exit_index,0);
    
    MetalSparseMidpoint midpoint2{inputs[1], 1e-5, device, 2};

    
//    flk(inputs[2],temp);
//    auto [success_flag2,exit_index2,time_step2] = fek(inputs[2], midpoint2, temp);
//    std::cout << "exit_index2 = " <<exit_index2 << "\n";
}
//
////#include "simplex_tree.hpp"
////#include "debug_display.hpp"
////#include "random_generators.hpp"
////#include "facing_node.hpp"
////#include "numerical_debug.hpp"
////
////int main(){
////    int d = int(1e2);
////    SimplexTree st(0, 1e-9);
////    st.midpoint_table.max_dim = d;
////    InterpolationTest intest{&st};
////    srand(1);
////    for(int i = 0; i <16; i ++ ){
////        std::vector<float> x{};
////        random_float_vector(&x, d);
////        auto alias = st.add_midpoint(x);
////        if(alias == std::numeric_limits<size_t>::max()){
////            continue;
////        }
////        std::cout << "#node = " << st.midpoint_table.data.size() <<", ";
////        std::cout << "#midpoint = " << st.midpoint_table.midpoints.size() << ", ";
////        std::cout << " depth = " <<
////                    st.midpoint_table.depth << "\n";
////        log_sink.report();
////        std::cout << " --------------- " << "\n";
////    }
////    for(int i = 0; i < 12; i++){
////        std::vector<float> x{};
////        random_float_vector(&x, d);
////        auto err = intest(x);
////        std::cout << "test # " << i << " err = ";
////        sciprint(err,2);
////        std::cout << "\n";
////    }
////}
//
