//
//  ops.metal
//  LinearInterpolation
//
//  Created by Cem Gultekin on 5/9/24.
//

#include <metal_stdlib>

using namespace metal;





struct Sum {
  template <typename T, typename U>
  inline T operator()(thread const T& a, thread const U& b) const {
    return a + b;
  }
  template <typename T, typename U>
  inline T operator()(threadgroup const T& a,
                      threadgroup const U& b) const {
    return a + b;
  }
};

struct Max {
  template <typename T, typename U>
  inline T operator()(thread const T& a, thread const U& b) const {
    return fmax(a,b);
  }
  template <typename T, typename U>
  inline T operator()(threadgroup const T& a,
                      threadgroup const U& b) const {
    return fmax(a,b);
  }
};




//////// REDUCTION LIBRARY//////////////

template<typename T, ushort PER_THREAD, typename OPERATION>
static void LoadLocalReduceFromGlobal(thread T &value,
                                      const device T* input_data,
                                      const uint lid) {
    OPERATION Op;
    switch(PER_THREAD){
        case 1:{
            value = Op(value,input_data[lid * PER_THREAD]);
            break;
        }
        case 2:{
            T a1 = input_data[lid * PER_THREAD];
            T a2 = input_data[lid * PER_THREAD+1];
            value = Op(Op(a1,a2),value);
            break;
        }
        case 4:{
            T a1 = input_data[lid * PER_THREAD];
            T a2 = input_data[lid * PER_THREAD+1];
            T a3 = input_data[lid * PER_THREAD+2];
            T a4 = input_data[lid * PER_THREAD+3];
            value = Op(Op(Op(a1,a2),Op(a3,a4)),value);
            break;
        }
        case 8:{
            T a1 = input_data[lid * PER_THREAD];
            T a2 = input_data[lid * PER_THREAD+1];
            T a3 = input_data[lid * PER_THREAD+2];
            T a4 = input_data[lid * PER_THREAD+3];
            T a5 = input_data[lid * PER_THREAD+4];
            T a6 = input_data[lid * PER_THREAD+5];
            T a7 = input_data[lid * PER_THREAD+6];
            T a8 = input_data[lid * PER_THREAD+7];
            a1 = Op(a1,a2);
            a2 = Op(a3,a4);
            a3 = Op(a5,a6);
            a4 = Op(a7,a8);
            a1 = Op(a1,a2);
            a2 = Op(a3,a4);
            a1 = Op(a1,a2);
            value = Op(a1,value);
            break;
        }
        default:{
            for (ushort i = 0; i < PER_THREAD; i++){
                value = Op(value,input_data[lid * PER_THREAD+i]);
            }
            break;
        }
    }
  
}


template <typename T,typename OPERATION>
static T ThreadgroupReduceSharedMemAlgorithm(
                                    thread T& value,
                                    threadgroup T* shared,
                                    const uint lid,
                                    const uint threadgroup_size,
                                    const uint execution_width) {
    // copy values to shared memory
    shared[lid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    OPERATION Op;
    // raking
    if (lid < execution_width) {
        for(ushort i = 1; i< threadgroup_size/execution_width; i++){
            shared[lid] = Op(shared[lid], shared[lid + i*execution_width]);
        }
        value = shared[lid];
//        if(threadgroup_size > execution_width){
        for(ushort i = execution_width/2; i > 0; i/=2){
            value = Op(value,simd_shuffle_down(value, i));
        }
//        }
        
    }
    return value;
}

template<typename T,ushort PER_THREAD,typename OPERATION>
static void full_reduction(
                        device T* X,
                        device T* Y,
                        threadgroup T* scratch,
                        const uint thread_position_in_grid            [[thread_position_in_grid]],
                        const uint thread_index_in_threadgroup        [[thread_index_in_threadgroup]],
                        const uint threads_per_threadgroup            [[threads_per_threadgroup]],
                        const uint execution_width                    [[threads_per_simdgroup]],
                        const uint threadgroup_position_in_grid       [[threadgroup_position_in_grid]])
{

    T value = 0.;
    LoadLocalReduceFromGlobal<T,PER_THREAD,OPERATION>(value,X,thread_position_in_grid);
    value =ThreadgroupReduceSharedMemAlgorithm<T,OPERATION>(value,scratch,thread_index_in_threadgroup,threads_per_threadgroup,execution_width);
    if(thread_index_in_threadgroup == 0){
        Y[threadgroup_position_in_grid] = value;
    }
}

//////////FIRST LAYER KERNEL/////////

template<typename T, ushort PER_THREAD> kernel void
first_layer_template(
                                 device T* X,
                                 device T* temp,
                                 threadgroup T* scratch                        [[ threadgroup(0) ]],
                                 const uint thread_position_in_grid            [[thread_position_in_grid]],
                                 const uint thread_index_in_threadgroup        [[thread_index_in_threadgroup]],
                                 const uint threads_per_threadgroup            [[threads_per_threadgroup]],
                                 const uint execution_width                    [[threads_per_simdgroup]],
                                 const uint num_threadgroups                   [[threadgroups_per_grid]],
                                 const uint threadgroup_position_in_grid       [[threadgroup_position_in_grid]]){
     T n = num_threadgroups*threads_per_threadgroup*PER_THREAD - 1;
     for (ushort i = 0; i < PER_THREAD; i++){
         X[thread_position_in_grid*PER_THREAD + i] /= n;
     }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    full_reduction<T, PER_THREAD,Sum>(X, temp, scratch, thread_position_in_grid, thread_index_in_threadgroup, threads_per_threadgroup, execution_width,threadgroup_position_in_grid);
    
}



template [[host_name("first_layer_pt_1")]]  kernel void first_layer_template<float,1>(device float*, device float*,threadgroup float*,const uint,const uint,const uint,const uint,const uint,const uint);
template [[host_name("first_layer_pt_2")]]  kernel void first_layer_template<float,2>(device float*, device float*,threadgroup float*,const uint,const uint,const uint,const uint,const uint,const uint);
template [[host_name("first_layer_pt_4")]]  kernel void first_layer_template<float,4>(device float*, device float*,threadgroup float*,const uint,const uint,const uint,const uint,const uint,const uint);
template [[host_name("first_layer_pt_8")]]  kernel void first_layer_template<float,8>(device float*, device float*,threadgroup float*,const uint,const uint,const uint,const uint,const uint,const uint);
template [[host_name("first_layer_pt_16")]]  kernel void first_layer_template<float,16>(device float*, device float*,threadgroup float*,const uint,const uint,const uint,const uint,const uint,const uint);
template [[host_name("first_layer_pt_32")]]  kernel void first_layer_template<float,32>(device float*, device float*,threadgroup float*,const uint,const uint,const uint,const uint,const uint,const uint);
template [[host_name("first_layer_pt_64")]]  kernel void first_layer_template<float,64>(device float*, device float*,threadgroup float*,const uint,const uint,const uint,const uint,const uint,const uint);
template [[host_name("first_layer_pt_128")]]  kernel void first_layer_template<float,128>(device float*, device float*,threadgroup float*,const uint,const uint,const uint,const uint,const uint,const uint);
template [[host_name("first_layer_pt_256")]]  kernel void first_layer_template<float,256>(device float*, device float*,threadgroup float*,const uint,const uint,const uint,const uint,const uint,const uint);
template [[host_name("first_layer_pt_512")]]  kernel void first_layer_template<float,512>(device float*, device float*,threadgroup float*,const uint,const uint,const uint,const uint,const uint,const uint);
template [[host_name("first_layer_pt_1024")]]  kernel void first_layer_template<float,1024>(device float*, device float*,threadgroup float*,const uint,const uint,const uint,const uint,const uint,const uint);

//
////////////SPARSIFICATION  KERNELS///////////
//
//
//template<typename T, ushort PER_THREAD> kernel void
//max_operation_template(
//                                 device T* X,
//                                 device T* temp,
//                                 const constant uint& n,
//                                 threadgroup T* scratch                        [[ threadgroup(0) ]],
//                                 const uint thread_position_in_grid            [[thread_position_in_grid]],
//                                 const uint thread_index_in_threadgroup        [[thread_index_in_threadgroup]],
//                                 const uint threads_per_threadgroup            [[threads_per_threadgroup]],
//                                 const uint execution_width                    [[threads_per_simdgroup]],
//                                 const uint num_threadgroups                   [[threadgroups_per_grid]],
//                                 const uint threadgroup_position_in_grid       [[threadgroup_position_in_grid]]){
//    full_reduction<T, PER_THREAD,Max>(X, temp, scratch, thread_position_in_grid, thread_index_in_threadgroup, threads_per_threadgroup, execution_width,threadgroup_position_in_grid);
//}
//
//
//template [[host_name("max_operation_float_pt_1")]]  kernel void max_operation_template<float,1>(device float*, device float*,const constant uint&,threadgroup float*,const uint,const uint,const uint,const uint,const uint,const uint);
//template [[host_name("max_operation_float_pt_2")]]  kernel void max_operation_template<float,2>(device float*, device float*,const constant uint&,threadgroup float*,const uint,const uint,const uint,const uint,const uint,const uint);
//template [[host_name("max_operation_float_pt_4")]]  kernel void max_operation_template<float,4>(device float*, device float*,const constant uint&,threadgroup float*,const uint,const uint,const uint,const uint,const uint,const uint);
//template [[host_name("max_operation_float_pt_8")]]  kernel void max_operation_template<float,8>(device float*, device float*,const constant uint&,threadgroup float*,const uint,const uint,const uint,const uint,const uint,const uint);
//template [[host_name("max_operation_float_pt_16")]]  kernel void max_operation_template<float,16>(device float*, device float*,const constant uint&,threadgroup float*,const uint,const uint,const uint,const uint,const uint,const uint);
//template [[host_name("max_operation_float_pt_32")]]  kernel void max_operation_template<float,32>(device float*, device float*,const constant uint&,threadgroup float*,const uint,const uint,const uint,const uint,const uint,const uint);
//template [[host_name("max_operation_float_pt_64")]]  kernel void max_operation_template<float,64>(device float*, device float*,const constant uint&,threadgroup float*,const uint,const uint,const uint,const uint,const uint,const uint);
//template [[host_name("max_operation_float_pt_128")]]  kernel void max_operation_template<float,128>(device float*, device float*,const constant uint&,threadgroup float*,const uint,const uint,const uint,const uint,const uint,const uint);
//template [[host_name("max_operation_float_pt_256")]]  kernel void max_operation_template<float,256>(device float*, device float*,const constant uint&,threadgroup float*,const uint,const uint,const uint,const uint,const uint,const uint);
//template [[host_name("max_operation_float_pt_512")]]  kernel void max_operation_template<float,512>(device float*, device float*,const constant uint&,threadgroup float*,const uint,const uint,const uint,const uint,const uint,const uint);
//template [[host_name("max_operation_float_pt_1024")]]  kernel void max_operation_template<float,1024>(device float*, device float*,const constant uint&,threadgroup float*,const uint,const uint,const uint,const uint,const uint,const uint);
//
//
//
//template<typename T,typename I, ushort PER_THREAD> kernel void
//sparsification_I_template(
//device T* X,
//device I* glb_nnz,
//device T* glb_sum,
//const constant T& lolim,
//threadgroup I* nnz                        [[ threadgroup(0) ]],
//threadgroup T* sum                        [[ threadgroup(1) ]],
//const uint thread_position_in_grid            [[thread_position_in_grid]],
//const uint lid                              [[thread_index_in_threadgroup]],
//const uint threads_per_threadgroup            [[threads_per_threadgroup]],
//const uint execution_width                    [[threads_per_simdgroup]],
//const uint num_threadgroups                   [[threadgroups_per_grid]],
//const uint gid       [[threadgroup_position_in_grid]]){
//    I local_nnz = 0;
//    T local_sum = 0;
//    for(uint i = 0; i < PER_THREAD;++i){
//        bool flag = X[gid*PER_THREAD + i] > lolim;
//        local_nnz += flag ? 1 : 0;
//        local_sum += flag ? X[gid*PER_THREAD + i] : 0;
//    }
//    local_nnz = ThreadgroupReduceSharedMemAlgorithm<I,Sum>(local_nnz,nnz,lid,threads_per_threadgroup,execution_width);
//    local_sum = ThreadgroupReduceSharedMemAlgorithm<T,Sum>(local_sum,sum,lid,threads_per_threadgroup,execution_width);
//    if(lid == 0){
//        glb_nnz[lid] = local_nnz;
//        glb_sum[lid] = local_sum;
//    }
//}
//
//
//template<typename T,typename I, ushort PER_THREAD> kernel void
//sparsification_II_template(
//device T* X,
//device I* glb_nnz,
//device T* glb_sum,
//const constant T& lolim,
//threadgroup I* nnz                        [[ threadgroup(0) ]],
//threadgroup T* sum                        [[ threadgroup(1) ]],
//const uint thread_position_in_grid            [[thread_position_in_grid]],
//const uint lid                              [[thread_index_in_threadgroup]],
//const uint threads_per_threadgroup            [[threads_per_threadgroup]],
//const uint execution_width                    [[threads_per_simdgroup]],
//const uint num_threadgroups                   [[threadgroups_per_grid]],
//const uint gid       [[threadgroup_position_in_grid]]){
//    I local_nnz = 0;
//    T local_sum = 0;
//    for(uint i = 0; i < PER_THREAD;++i){
//        bool flag = X[gid*PER_THREAD + i] > lolim;
//        local_nnz += flag ? 1 : 0;
//        local_sum += flag ? X[gid*PER_THREAD + i] : 0;
//    }
//    local_nnz = ThreadgroupReduceSharedMemAlgorithm<I,Sum>(local_nnz,nnz,lid,threads_per_threadgroup,execution_width);
//    local_sum = ThreadgroupReduceSharedMemAlgorithm<T,Sum>(local_sum,sum,lid,threads_per_threadgroup,execution_width);
//    if(lid == 0){
//        glb_nnz[lid] = local_nnz;
//        glb_sum[lid] = local_sum;
//    }
//}
