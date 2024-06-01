//
//  copy_kernel.metal
//  LinearInterpolation
//
//  Created by Cem Gultekin on 5/21/24.
//

#include <metal_stdlib>
using namespace metal;




template<typename T, ushort PER_THREAD>
kernel void copy_template(
device T* X,
device T* temp,
const uint gid [[threadgroup_position_in_grid]]){
    for(uint i = 0 ; i < PER_THREAD; ++i){
        temp[gid*PER_THREAD+i] = X[gid*PER_THREAD+i];
    }
}


template [[host_name("copy_float_pt_1")]]  kernel void copy_template<float,1>(device float*, device float*,const uint);

template [[host_name("copy_float_pt_2")]]  kernel void copy_template<float,2>(device float*, device float*,const uint);

template [[host_name("copy_float_pt_4")]]  kernel void copy_template<float,4>(device float*, device float*,const uint);

template [[host_name("copy_float_pt_8")]]  kernel void copy_template<float,8>(device float*, device float*,const uint);

template [[host_name("copy_float_pt_16")]]  kernel void copy_template<float,16>(device float*, device float*,const uint);

template [[host_name("copy_float_pt_32")]]  kernel void copy_template<float,32>(device float*, device float*,const uint);


