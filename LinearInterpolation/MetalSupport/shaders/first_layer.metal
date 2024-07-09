//
//  ops.metal
//  LinearInterpolation
//
//  Created by Cem Gultekin on 5/9/24.
//

#include <metal_stdlib>
#include "macros.h"
#include "../../Utils/formulas.h"


using namespace metal;

//////// REDUCTION LIBRARY//////////////

template<ushort PER_THREAD>
static void LoadLocalReduceFromGlobal(thread float &value,
                                      const device float* input_data,
                                      const uint lid) {
    for (ushort i = 0; i < PER_THREAD; i++){
        value  += input_data[lid * PER_THREAD+i];
    }
}


static float ThreadgroupReduceSharedMemAlgorithm(
                                    thread float& value,
                                    threadgroup float* shared,
                                    const uint lid,
                                    const uint threadgroup_size,
                                    const uint execution_width) {
    // copy values to shared memory
    shared[lid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // raking
    if (lid < execution_width) {
        for(ushort i = 1; i< threadgroup_size/execution_width; i++){
            shared[lid]  += shared[lid + i*execution_width];
        }
        value = shared[lid];
//        if(threadgroup_size > execution_width){
        for(ushort i = execution_width/2; i > 0; i/=2){
            value += simd_shuffle_down(value, i);
        }
//        }
        
    }
    return value;
}

template<ushort PER_THREAD>
static void full_reduction(
                        device float* X,
                        device float* Y,
                        threadgroup float* scratch,
                        const uint thread_position_in_grid            [[thread_position_in_grid]],
                        const uint thread_index_in_threadgroup        [[thread_index_in_threadgroup]],
                        const uint threads_per_threadgroup            [[threads_per_threadgroup]],
                        const uint execution_width                    [[threads_per_simdgroup]],
                        const uint threadgroup_position_in_grid       [[threadgroup_position_in_grid]])
{

    float value = 0.;
    LoadLocalReduceFromGlobal<PER_THREAD>(value,X,thread_position_in_grid);
    value=ThreadgroupReduceSharedMemAlgorithm(value,scratch,thread_index_in_threadgroup,threads_per_threadgroup,execution_width);
    if(thread_index_in_threadgroup == 0){
        Y[threadgroup_position_in_grid] = value;
    }
}

#define CUR_KRNL FIRST_LAYER_KRNL_NM

#define CUR_ARGS \
 device float* X,\
 device float* temp,\
 constant float& _min,\
 constant float& _max,\
 constant float& _mean,\
 constant float& dim,\
 threadgroup float* scratch                    [[ threadgroup(0) ]],\
 const uint thread_position_in_grid            [[thread_position_in_grid]],\
 const uint thread_index_in_threadgroup        [[thread_index_in_threadgroup]],\
 const uint threads_per_threadgroup            [[threads_per_threadgroup]],\
 const uint execution_width                    [[threads_per_simdgroup]],\
 const uint num_threadgroups                   [[threadgroups_per_grid]],\
 const uint threadgroup_position_in_grid       [[threadgroup_position_in_grid]]
KRNL_HDR(CUR_KRNL,CUR_ARGS){
    
    if(_max != 1 || _min != 0){
        for (ushort i = 0; i < PER_THREAD; i++){
            X[thread_position_in_grid*PER_THREAD + i] = (X[thread_position_in_grid*PER_THREAD + i] - _min)/(_max - _min)/dim;
        }
    }else{
        for (ushort i = 0; i < PER_THREAD; i++){
            X[thread_position_in_grid*PER_THREAD + i] = X[thread_position_in_grid*PER_THREAD + i]/dim;
        }
    }
     
    threadgroup_barrier(mem_flags::mem_threadgroup);
    full_reduction<PER_THREAD>(X, temp, scratch, thread_position_in_grid, thread_index_in_threadgroup, threads_per_threadgroup, execution_width,threadgroup_position_in_grid);
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(num_threadgroups == 1){
        float _mean_hat = temp[threadgroup_position_in_grid];
        float kappa = kappa_fun(_mean, dim);
        float beta = beta_fun(_mean_hat, dim, kappa);
        for (ushort i = 0; i < PER_THREAD; i++){
            X[thread_position_in_grid*PER_THREAD + i] += beta;
        }
        if(thread_position_in_grid + 1 == threads_per_threadgroup){
            X[(thread_position_in_grid+1)*PER_THREAD - 1] = last_entry_fun(_mean_hat, kappa);
        }
    }
}


HST_DCL(CUR_KRNL,CUR_ARGS,1);
HST_DCL(CUR_KRNL,CUR_ARGS,2);
HST_DCL(CUR_KRNL,CUR_ARGS,4);
HST_DCL(CUR_KRNL,CUR_ARGS,8);
HST_DCL(CUR_KRNL,CUR_ARGS,16);
HST_DCL(CUR_KRNL,CUR_ARGS,32);
HST_DCL(CUR_KRNL,CUR_ARGS,64);
HST_DCL(CUR_KRNL,CUR_ARGS,128);
HST_DCL(CUR_KRNL,CUR_ARGS,256);
HST_DCL(CUR_KRNL,CUR_ARGS,512);
HST_DCL(CUR_KRNL,CUR_ARGS,1024);
HST_DCL(CUR_KRNL,CUR_ARGS,2048);
HST_DCL(CUR_KRNL,CUR_ARGS,4096);
HST_DCL(CUR_KRNL,CUR_ARGS,8192);
HST_DCL(CUR_KRNL,CUR_ARGS,16384);







#define CUR_KRNL FIRST_LAYER_PSTR_KRNL_NM

#define CUR_ARGS \
 device float* X,\
 constant float& mean,\
 constant float& dim,\
 const uint thread_position_in_grid            [[thread_position_in_grid]],\
 const uint thread_index_in_threadgroup        [[thread_index_in_threadgroup]],\
 const uint threads_per_threadgroup            [[threads_per_threadgroup]],\
 const uint execution_width                    [[threads_per_simdgroup]],\
 const uint num_threadgroups                   [[threadgroups_per_grid]],\
 const uint threadgroup_position_in_grid       [[threadgroup_position_in_grid]]

KRNL_HDR(CUR_KRNL,CUR_ARGS){
    float kappa = kappa_fun(mean,dim);
    float beta =  beta_fun(mean, dim, kappa);
    for (ushort i = 0; i < PER_THREAD; i++){
        X[thread_position_in_grid*PER_THREAD + i] += beta;
    }
    if(thread_position_in_grid + 1 == num_threadgroups*threads_per_threadgroup){
        X[(thread_position_in_grid+1)*PER_THREAD - 1] = last_entry_fun(mean, kappa);
    }
}

HST_DCL(CUR_KRNL,CUR_ARGS,1);
HST_DCL(CUR_KRNL,CUR_ARGS,2);
HST_DCL(CUR_KRNL,CUR_ARGS,4);
HST_DCL(CUR_KRNL,CUR_ARGS,8);
HST_DCL(CUR_KRNL,CUR_ARGS,16);
HST_DCL(CUR_KRNL,CUR_ARGS,32);
HST_DCL(CUR_KRNL,CUR_ARGS,64);
HST_DCL(CUR_KRNL,CUR_ARGS,128);
HST_DCL(CUR_KRNL,CUR_ARGS,256);
HST_DCL(CUR_KRNL,CUR_ARGS,512);
HST_DCL(CUR_KRNL,CUR_ARGS,1024);
HST_DCL(CUR_KRNL,CUR_ARGS,2048);
HST_DCL(CUR_KRNL,CUR_ARGS,4096);
HST_DCL(CUR_KRNL,CUR_ARGS,8192);
HST_DCL(CUR_KRNL,CUR_ARGS,16384);

