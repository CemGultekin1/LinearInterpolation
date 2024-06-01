//
//  midpoint_kernels.metal
//  LinearInterpolation
//
//  Created by Cem Gultekin on 5/18/24.
//

#include <metal_stdlib>
using namespace metal;

#ifndef _NODE_TYPE
#define _NODE_TYPE uint
#endif

#ifndef _VALUE_TYPE
#define _VALUE_TYPE float
#endif


////////// MIDPOINT  KERNELS ///////////
template<typename I,typename T>
static void keepMin(thread char& legal,thread I&local_min_index, thread T& local_min_value,
                const device T* x, const constant I& x_origin,const device I* midpoint_node,
                    const device T* midpoint_value,const uint lid,const uint n){
    I index = midpoint_node[lid%n] + x_origin;
    T xw =  x[index >= 0 ? index : 0];
    T mw = midpoint_value[lid%n];
    bool in_bound = lid < n;
    mw = xw/mw;
    legal &= (char) ((xw > 0) && (index >= 0) || !in_bound);
    local_min_index =  (local_min_value > mw && in_bound) ? lid : local_min_index;
    local_min_value = (local_min_value > mw && in_bound) ? mw   : local_min_value;
}

template<typename I,typename T, ushort PER_THREAD>
static void LoadLocalReduceFromGlobal(thread char& legal,
                                      thread I&local_min_index,
                                      thread T&local_min_value,
                                      const device T* x,
                                      const constant I& x_origin,
                                      const device I* midpoint_node,
                                      const device T* midpoint_value,
                                      const uint gid,
                                      const uint n) {
    for (uint i = 0; i < PER_THREAD; i++){
        uint j = gid*PER_THREAD+i;
        keepMin(legal,local_min_index,local_min_value,x,x_origin,midpoint_node,midpoint_value,j,n);
    }
}

template <typename I,typename T>
T assign_min_value(const thread I& node1,const thread T& value1,const thread I& node2,const thread T& value2){
    return (value1 < value2) ? value1 : value2;
}

template <typename I,typename T>
T assign_min_value(const threadgroup I& node1,const threadgroup T& value1,const threadgroup I& node2,const threadgroup T& value2){
    return (value1 < value2) ? value1 : value2;
}

template <typename I,typename T>
I assign_min_node(const thread I& node1,const thread T& value1,const thread I& node2,const thread T& value2){
    return (value1 < value2) ? node1 : node2;
}

template <typename I,typename T>
I assign_min_node(const threadgroup I& node1,const threadgroup T& value1,const threadgroup I& node2,const threadgroup T& value2){
    return (value1 < value2) ? node1 : node2;
}


template <typename I,typename T>
static void ThreadgroupReduceSharedMemAlgorithm(
thread char& legal,
thread I&local_min_index,
thread T&local_min_value,
threadgroup char* legal_shared,
threadgroup I* min_shared_node,
threadgroup T* min_shared_value,
const uint lid,
const uint threadgroup_size,
const uint execution_width) {
    // copy values to shared memory
    min_shared_value[lid] = local_min_value;
    min_shared_node[lid] = local_min_index;
    legal_shared[lid] = legal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // raking
    if (lid < execution_width) {
        for(ushort i = 1; i< threadgroup_size/execution_width; i++){
            min_shared_node[lid] = assign_min_node(
                             min_shared_node[lid],
                             min_shared_value[lid],
                             min_shared_node[lid + i*execution_width],
                             min_shared_value[lid + i*execution_width]);
            min_shared_value[lid] = assign_min_value(
                             min_shared_node[lid],
                             min_shared_value[lid],
                             min_shared_node[lid + i*execution_width],
                             min_shared_value[lid + i*execution_width]);
            legal_shared[lid] &= legal_shared[lid + i*execution_width];
        }
        local_min_index = min_shared_node[lid];
        local_min_value = min_shared_value[lid];
        legal = legal_shared[lid];
        for(ushort i = execution_width/2; i > 0; i/=2){
            I local_node_up = simd_shuffle_down(local_min_index, i);
            T local_value_up = simd_shuffle_down(local_min_value, i);
            local_min_index = assign_min_node(local_min_index,local_min_value,local_node_up,local_value_up);
            local_min_value = assign_min_value(local_min_index,local_min_value,local_node_up,local_value_up);
            legal &= simd_shuffle_down(legal, i);
        }
    }
}

template<typename I,typename T, ushort PER_THREAD> kernel void
find_midpoint_exit_node_template(
const device T* x,
const constant I& x_origin,
const device I* midpoint_node,
const device T* midpoint_value,
const constant uint& n,
device I* glbl_max_index,
device T* glbl_max_value,
device char* glbl_legal,
threadgroup I* local_min_shared_node            [[ threadgroup(0)]],
threadgroup T* local_min_shared_value           [[ threadgroup(1)]],
threadgroup char* legal_shared                  [[ threadgroup(2) ]],
const uint thread_position_in_grid            [[thread_position_in_grid]],
const uint thread_index_in_threadgroup        [[thread_index_in_threadgroup]],
const uint threads_per_threadgroup            [[threads_per_threadgroup]],
const uint execution_width                    [[threads_per_simdgroup]],
const uint num_threadgroups                   [[threadgroups_per_grid]],
const uint threadgroup_position_in_grid       [[threadgroup_position_in_grid]]){
    legal_shared[threads_per_threadgroup] = 1;
    char legal = 1;
    I local_min_index = 0;
    T local_min_value = 1.;
    LoadLocalReduceFromGlobal<I,T,PER_THREAD>(legal,local_min_index,local_min_value,x,x_origin,midpoint_node,midpoint_value,thread_position_in_grid,n);
    ThreadgroupReduceSharedMemAlgorithm<I,T>(legal,local_min_index,local_min_value,
                                             legal_shared,local_min_shared_node,local_min_shared_value,
                                             thread_index_in_threadgroup,threads_per_threadgroup,execution_width);
    if(thread_index_in_threadgroup == 0){
        glbl_legal[threadgroup_position_in_grid] = legal;
        glbl_max_index[threadgroup_position_in_grid] = local_min_index;
        glbl_max_value[threadgroup_position_in_grid] = local_min_value;
    }
    
}


template [[host_name("find_midpoint_exit_node_pt_1")]]  kernel void
find_midpoint_exit_node_template<_NODE_TYPE,_VALUE_TYPE,1>(
const device _VALUE_TYPE* ,const constant _NODE_TYPE&, const device _NODE_TYPE* , const device _VALUE_TYPE*,const constant uint&,
device _NODE_TYPE*, device _VALUE_TYPE*, device char*,
threadgroup _NODE_TYPE*, threadgroup _VALUE_TYPE* ,threadgroup char* ,const uint,const uint,const uint,const uint,const uint,const uint);


template [[host_name("find_midpoint_exit_node_pt_2")]]  kernel void
find_midpoint_exit_node_template<_NODE_TYPE,_VALUE_TYPE,2>(
const device _VALUE_TYPE* ,const constant _NODE_TYPE&, const device _NODE_TYPE* , const device _VALUE_TYPE*,const constant uint&,
device _NODE_TYPE*, device _VALUE_TYPE*, device char*,
threadgroup _NODE_TYPE*, threadgroup _VALUE_TYPE* ,threadgroup char* ,const uint,
const uint,const uint,const uint,const uint,const uint);



template [[host_name("find_midpoint_exit_node_pt_4")]]  kernel void
find_midpoint_exit_node_template<_NODE_TYPE,_VALUE_TYPE,4>(
const device _VALUE_TYPE* ,const constant _NODE_TYPE&, const device _NODE_TYPE* , const device _VALUE_TYPE*,const constant uint&,
device _NODE_TYPE*, device _VALUE_TYPE*, device char*,
threadgroup _NODE_TYPE*, threadgroup _VALUE_TYPE* ,threadgroup char* ,const uint,
const uint,const uint,const uint,const uint,const uint);



template [[host_name("find_midpoint_exit_node_pt_8")]]  kernel void
find_midpoint_exit_node_template<_NODE_TYPE,_VALUE_TYPE,8>(const device _VALUE_TYPE* ,const constant _NODE_TYPE&, const device _NODE_TYPE* , const device _VALUE_TYPE*,const constant uint&,
device _NODE_TYPE*, device _VALUE_TYPE*, device char*,
threadgroup _NODE_TYPE*, threadgroup _VALUE_TYPE* ,threadgroup char* ,const uint,
const uint,const uint,const uint,const uint,const uint);



template [[host_name("find_midpoint_exit_node_pt_16")]]  kernel void
find_midpoint_exit_node_template<_NODE_TYPE,_VALUE_TYPE,16>(
const device _VALUE_TYPE* ,const constant _NODE_TYPE&, const device     _NODE_TYPE* , const device _VALUE_TYPE*,const constant uint&,
device _NODE_TYPE*, device _VALUE_TYPE*, device char*,
threadgroup _NODE_TYPE*, threadgroup _VALUE_TYPE* ,threadgroup char* ,const uint,
const uint,const uint,const uint,const uint,const uint);



template [[host_name("find_midpoint_exit_node_pt_32")]]  kernel void
find_midpoint_exit_node_template<_NODE_TYPE,_VALUE_TYPE,32>(
const device _VALUE_TYPE* ,const constant _NODE_TYPE&, const device _NODE_TYPE* , const device _VALUE_TYPE*,const constant uint&,device _NODE_TYPE*, device _VALUE_TYPE*, device char*,
threadgroup _NODE_TYPE*, threadgroup _VALUE_TYPE* ,threadgroup char* ,const uint,
const uint,const uint,const uint,const uint,const uint);

template [[host_name("find_midpoint_exit_node_pt_64")]]  kernel void
find_midpoint_exit_node_template<_NODE_TYPE,_VALUE_TYPE,64>(
const device _VALUE_TYPE* ,const constant _NODE_TYPE&, const device _NODE_TYPE* , const device _VALUE_TYPE*,const constant uint&,device _NODE_TYPE*, device _VALUE_TYPE*, device char*,
threadgroup _NODE_TYPE*, threadgroup _VALUE_TYPE* ,threadgroup char* ,const uint,
const uint,const uint,const uint,const uint,const uint);

template [[host_name("find_midpoint_exit_node_pt_128")]]  kernel void
find_midpoint_exit_node_template<_NODE_TYPE,_VALUE_TYPE,128>(
const device _VALUE_TYPE* ,const constant _NODE_TYPE&, const device _NODE_TYPE* , const device _VALUE_TYPE*,const constant uint&,device _NODE_TYPE*, device _VALUE_TYPE*, device char*,
threadgroup _NODE_TYPE*, threadgroup _VALUE_TYPE* ,threadgroup char* ,const uint,
const uint,const uint,const uint,const uint,const uint);

template [[host_name("find_midpoint_exit_node_pt_256")]]  kernel void
find_midpoint_exit_node_template<_NODE_TYPE,_VALUE_TYPE,256>(
const device _VALUE_TYPE* ,const constant _NODE_TYPE&, const device _NODE_TYPE* , const device _VALUE_TYPE*,const constant uint&,device _NODE_TYPE*, device _VALUE_TYPE*, device char*,
threadgroup _NODE_TYPE*, threadgroup _VALUE_TYPE* ,threadgroup char* ,const uint,
const uint,const uint,const uint,const uint,const uint);

template [[host_name("find_midpoint_exit_node_pt_512")]]  kernel void
find_midpoint_exit_node_template<_NODE_TYPE,_VALUE_TYPE,512>(
const device _VALUE_TYPE* ,const constant _NODE_TYPE&, const device _NODE_TYPE* , const device _VALUE_TYPE*,const constant uint&,device _NODE_TYPE*, device _VALUE_TYPE*, device char*,
threadgroup _NODE_TYPE*, threadgroup _VALUE_TYPE* ,threadgroup char* ,const uint,
const uint,const uint,const uint,const uint,const uint);

template [[host_name("find_midpoint_exit_node_pt_1024")]]  kernel void
find_midpoint_exit_node_template<_NODE_TYPE,_VALUE_TYPE,1024>(
const device _VALUE_TYPE* ,const constant _NODE_TYPE&, const device _NODE_TYPE* , const device _VALUE_TYPE*,const constant uint&,device _NODE_TYPE*, device _VALUE_TYPE*, device char*,
threadgroup _NODE_TYPE*, threadgroup _VALUE_TYPE* ,threadgroup char* ,const uint,
const uint,const uint,const uint,const uint,const uint);





template<typename I,typename T, ushort PER_THREAD> kernel void
midpoint_operation_template(
device T* x,
const constant I& x_origin,
const device I* midpoint_nodes,
const device T* midpoint_values,
const constant T& step_size,
const constant I& escape_index,
const constant I& midpoint_node,
const constant uint& n,
const uint gid                                [[thread_position_in_grid]],
const uint thread_index_in_threadgroup        [[thread_index_in_threadgroup]],
const uint threads_per_threadgroup            [[threads_per_threadgroup]],
const uint execution_width                    [[threads_per_simdgroup]],
const uint num_threadgroups                   [[threadgroups_per_grid]],
const uint threadgroup_position_in_grid       [[threadgroup_position_in_grid]],
const uint grid_size                          [[grid_size]]){
    for(unsigned int i = 0 ; i < PER_THREAD; i ++){
        I j = gid*PER_THREAD + i;
        I index = x_origin + midpoint_nodes[j%n];
        T y = midpoint_values[j%n];
        y = (y*step_size)*(j < n);
        x[index] -= y;
    }
    if(escape_index/PER_THREAD == gid){
        I escape_node = midpoint_nodes[escape_index];
        x[escape_node + x_origin] = 0;
        x[midpoint_node + x_origin] = step_size;
    }
}


template [[host_name("midpoint_operation_pt_1")]]  kernel void
midpoint_operation_template<_NODE_TYPE,_VALUE_TYPE,1>(
device _VALUE_TYPE* ,const constant _NODE_TYPE&,
const device _NODE_TYPE* ,const device _VALUE_TYPE*,
const constant _VALUE_TYPE&,const constant _NODE_TYPE&,const constant _NODE_TYPE&,
const constant uint&,const uint,const uint,const uint,const uint,const uint,const uint,const uint);

template [[host_name("midpoint_operation_pt_2")]]  kernel void
midpoint_operation_template<_NODE_TYPE,_VALUE_TYPE,2>(
device _VALUE_TYPE* ,const constant _NODE_TYPE&,const device _NODE_TYPE* ,const device _VALUE_TYPE*,
const constant _VALUE_TYPE&,const constant _NODE_TYPE&,const constant _NODE_TYPE&,
const constant uint&,const uint,const uint,const uint,const uint,const uint,const uint,const uint);

template [[host_name("midpoint_operation_pt_4")]]  kernel void
midpoint_operation_template<_NODE_TYPE,_VALUE_TYPE,4>(
device _VALUE_TYPE* ,const constant _NODE_TYPE&,const device _NODE_TYPE* ,const device _VALUE_TYPE*,
const constant _VALUE_TYPE&,const constant _NODE_TYPE&,const constant _NODE_TYPE&,
const constant uint&,const uint,const uint,const uint,const uint,const uint,const uint,const uint);

template [[host_name("midpoint_operation_pt_8")]]  kernel void
midpoint_operation_template<_NODE_TYPE,_VALUE_TYPE,8>(
device _VALUE_TYPE* ,const constant _NODE_TYPE&,const device _NODE_TYPE* ,const device _VALUE_TYPE*,
const constant _VALUE_TYPE&,const constant _NODE_TYPE&,const constant _NODE_TYPE&,
const constant uint&,const uint,const uint,const uint,const uint,const uint,const uint,const uint);

template [[host_name("midpoint_operation_pt_16")]]  kernel void
midpoint_operation_template<_NODE_TYPE,_VALUE_TYPE,16>(
device _VALUE_TYPE* ,const constant _NODE_TYPE&,const device _NODE_TYPE* ,const device _VALUE_TYPE*,
const constant _VALUE_TYPE&,const constant _NODE_TYPE&,const constant _NODE_TYPE&,
const constant uint&,const uint,const uint,const uint,const uint,const uint,const uint,const uint);

template [[host_name("midpoint_operation_pt_32")]]  kernel void
midpoint_operation_template<_NODE_TYPE,_VALUE_TYPE,32>(
device _VALUE_TYPE* ,const constant _NODE_TYPE&,const device _NODE_TYPE* ,const device _VALUE_TYPE*,
const constant _VALUE_TYPE&,const constant _NODE_TYPE&,const constant _NODE_TYPE&,
const constant uint&,const uint,const uint,const uint,const uint,const uint,const uint,const uint);

template [[host_name("midpoint_operation_pt_64")]]  kernel void
midpoint_operation_template<_NODE_TYPE,_VALUE_TYPE,64>(
device _VALUE_TYPE* ,const constant _NODE_TYPE&,const device _NODE_TYPE* ,const device _VALUE_TYPE*,
const constant _VALUE_TYPE&,const constant _NODE_TYPE&,const constant _NODE_TYPE&,
const constant uint&,const uint,const uint,const uint,const uint,const uint,const uint,const uint);

template [[host_name("midpoint_operation_pt_128")]]  kernel void
midpoint_operation_template<_NODE_TYPE,_VALUE_TYPE,128>(
device _VALUE_TYPE* ,const constant _NODE_TYPE&,const device _NODE_TYPE* ,const device _VALUE_TYPE*,
const constant _VALUE_TYPE&,const constant _NODE_TYPE&,const constant _NODE_TYPE&,
const constant uint&,const uint,const uint,const uint,const uint,const uint,const uint,const uint);

template [[host_name("midpoint_operation_pt_256")]]  kernel void
midpoint_operation_template<_NODE_TYPE,_VALUE_TYPE,256>(
device _VALUE_TYPE* ,const constant _NODE_TYPE&,const device _NODE_TYPE* ,const device _VALUE_TYPE*,
const constant _VALUE_TYPE&,const constant _NODE_TYPE&,const constant _NODE_TYPE&,
const constant uint&,const uint,const uint,const uint,const uint,const uint,const uint,const uint);

template [[host_name("midpoint_operation_pt_512")]]  kernel void
midpoint_operation_template<_NODE_TYPE,_VALUE_TYPE,512>(
device _VALUE_TYPE* ,const constant _NODE_TYPE&,const device _NODE_TYPE* ,const device _VALUE_TYPE*,
const constant _VALUE_TYPE&,const constant _NODE_TYPE&,const constant _NODE_TYPE&,
const constant uint&,const uint,const uint,const uint,const uint,const uint,const uint,const uint);

template [[host_name("midpoint_operation_pt_1024")]]  kernel void
midpoint_operation_template<_NODE_TYPE,_VALUE_TYPE,1024>(
device _VALUE_TYPE* ,const constant _NODE_TYPE&,const device _NODE_TYPE* ,const device _VALUE_TYPE*,
const constant _VALUE_TYPE&,const constant _NODE_TYPE&,const constant _NODE_TYPE&,
const constant uint&,const uint,const uint,const uint,const uint,const uint,const uint,const uint);
