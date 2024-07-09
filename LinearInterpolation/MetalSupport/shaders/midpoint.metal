//
//  midpoint_kernels.metal
//  LinearInterpolation
//
//  Created by Cem Gultekin on 5/18/24.
//

#include <metal_stdlib>
#include "macros.h"
#include "../../Utils/formulas.h"
using namespace metal;


////////// MIDPOINT  KERNELS ///////////
static void keepMin(
                thread uint &local_min_index,
                thread float& local_min_value,
                const device float* x,
                const device uint* midpoint_nodes,
                const device float* midpoint_value,
                const uint lid){
    
    uint j = midpoint_nodes[lid];
    float mw = midpoint_value[lid];
    float xw =  x[j];
    mw = division_zero_permissible(xw,mw);
    local_min_index =  (local_min_value > mw ) ? lid : local_min_index;
    local_min_value =  (local_min_value > mw ) ? mw  : local_min_value;
}

template<ushort PER_THREAD>
static void LoadLocalReduceFromGlobal(thread uint&local_min_index,
                                      thread float&local_min_value,
                                      const device float* x,
                                      const device uint* midpoint_nodes,
                                      const device float* midpoint_value,
                                      const constant uint& n,
                                      const uint gid
                                      ) {
    for (uint i = 0; i < PER_THREAD; i++){
        uint j = gid*PER_THREAD+i;
        if(j >= n){
            break;
        }
        keepMin(local_min_index,local_min_value,x,midpoint_nodes,midpoint_value,j);
    }
}



void assign_min_value(
                    thread uint& index1,
                    thread float& value1,
                    thread uint& index2,
                    thread float& value2){
    index1 =  (value1 < value2) ? index1 : index2;
    value1 =  (value1 < value2) ? value1 : value2;
}


void assign_min_value(threadgroup uint& index1,
                      threadgroup float& value1,
                      threadgroup uint& index2,
                      threadgroup float& value2){
    index1 =  (value1 < value2) ? index1 : index2;
    value1 =  (value1 < value2) ? value1 : value2;
}




static void ThreadgroupReduceSharedMemAlgorithm(
thread uint&local_min_index,
thread float&local_min_value,
threadgroup uint* min_shared_node,
threadgroup float* min_shared_value,
const uint lid,
const uint threadgroup_size,
const uint execution_width) {
    // copy values to shared memory
    min_shared_value[lid] = local_min_value;
    min_shared_node[lid] = local_min_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // raking
    if (lid < execution_width) {
        for(ushort i = 1; i< threadgroup_size/execution_width; i++){
            assign_min_value(min_shared_node[lid],
                             min_shared_value[lid],
                             min_shared_node[lid + i*execution_width],
                             min_shared_value[lid + i*execution_width]);
        }
        local_min_index = min_shared_node[lid];
        local_min_value = min_shared_value[lid];
        for(ushort i = execution_width/2; i > 0; i/=2){
            uint local_node_up = simd_shuffle_down(local_min_index, i);
            float local_value_up = simd_shuffle_down(local_min_value, i);
            assign_min_value(local_min_index,local_min_value,local_node_up,local_value_up);
        }
    }
}



#define CUR_KRNL FIND_EXIT_KRNL_NM

#define CUR_ARGS \
    const device float* x,\
    const device uint* midpoint_node,\
    const device float* midpoint_value,\
    const constant uint& n,\
    device uint* glbl_max_index,\
    device float* glbl_max_value,\
    threadgroup uint* local_min_shared_node       [[ threadgroup(0)]],\
    threadgroup float* local_min_shared_value     [[ threadgroup(1)]],\
    const uint thread_position_in_grid            [[thread_position_in_grid]],\
    const uint thread_index_in_threadgroup        [[thread_index_in_threadgroup]],\
    const uint threads_per_threadgroup            [[threads_per_threadgroup]],\
    const uint execution_width                    [[threads_per_simdgroup]],\
    const uint num_threadgroups                   [[threadgroups_per_grid]],\
    const uint threadgroup_position_in_grid       [[threadgroup_position_in_grid]]


KRNL_HDR(CUR_KRNL,CUR_ARGS){
    uint local_min_index = 0;
    float local_min_value = MAXFLOAT;
    LoadLocalReduceFromGlobal<PER_THREAD>(local_min_index,local_min_value,x,midpoint_node,midpoint_value,n,thread_position_in_grid);
    
    ThreadgroupReduceSharedMemAlgorithm(local_min_index,
                                        local_min_value,
                                        local_min_shared_node,
                                        local_min_shared_value,
                                        thread_index_in_threadgroup,
                                        threads_per_threadgroup,execution_width);
    if(thread_index_in_threadgroup == 0){
        glbl_max_index[threadgroup_position_in_grid] = local_min_index;
        glbl_max_value[threadgroup_position_in_grid] = local_min_value;
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





#define CUR_KRNL MDPNT_OPR_KRNL_NM

#define CUR_ARGS \
device float* x,\
const device uint* midpoint_nodes,\
const device float* midpoint_values,\
const constant uint& n,\
const constant float& step_size,\
const constant uint& escape_index,\
const constant uint& midpoint_node,\
const uint gid                                [[thread_position_in_grid]],\
const uint thread_index_in_threadgroup        [[thread_index_in_threadgroup]],\
const uint threads_per_threadgroup            [[threads_per_threadgroup]],\
const uint execution_width                    [[threads_per_simdgroup]],\
const uint num_threadgroups                   [[threadgroups_per_grid]],\
const uint threadgroup_position_in_grid       [[threadgroup_position_in_grid]],\
const uint grid_size                          [[grid_size]]

KRNL_HDR(CUR_KRNL,CUR_ARGS){
    for(uint i = 0 ; i < PER_THREAD; i ++){
        uint j = gid*PER_THREAD + i;
        if(j < n){
            uint index = midpoint_nodes[j];
            float y = midpoint_values[j];
            y *= step_size;
            x[index] -= y;
        }else{
            break;
        }
    }
    if(escape_index/PER_THREAD == gid){
        uint escape_node = midpoint_nodes[escape_index];
        x[escape_node] = 0;
        x[midpoint_node] = step_size;
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
