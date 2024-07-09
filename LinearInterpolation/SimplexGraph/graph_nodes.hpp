//
//  point.hpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 6/5/24.
//

#ifndef graph_nodes_hpp
#define graph_nodes_hpp

#include <stdio.h>
#include <vector>
#include <unordered_map>
#include "numerical_ops.hpp"


//#include "metal_header.h"
//#include "MetalSupport/KernelLibrary.hpp"

//struct Device{
//private:
//    MTL::Device * mtl_device;
//public:
//    bool gpu() const;
//    bool cpu() const;
//    void to_cpu();
//    void to_gpu(MTL::Device * );
//};

//template <typename T>
//struct Buffer: public Device{
//private:
//    std::vector<T>* cpu_memory;
//    MetalBufferCutlet<T>* gpu_memory;
//public:
//    Buffer(std::vector<T>*,size_t size);
//};


//enum point_backend_configuration {dictionary_vector, vector};

//struct VectorPoint:public Device{
//    Buffer<float>* weights;
//    VectorPoint(std::vector<float>* data, size_t size);
//};

//struct DictionaryPoint:public Device{
//    Buffer<float>* weights;
//    std::unordered_map<long,size_t> node2index;
//    DictionaryPoint(std::vector<float>* data, size_t size);
//};

//struct IntermediatePoint:public Device{
//    point_backend_configuration config;
//    VectorPoint* vector_point;
//    DictionaryPoint* dictionary_point;
//    
//};


struct SparsePoint{
    std::vector<float> weights;
    std::vector<long> nodes;
    SparsePoint(const size_t max_dim);
    SparsePoint(const SparsePoint&);
    std::string to_string() const;
};

struct PointWithDictionary{
    std::vector<float> weights;
    std::unordered_map<long,size_t> node2index;
    PointWithDictionary(const std::vector<float>& input);
    PointWithDictionary(const size_t& max_dim);
    void to_point(SparsePoint & empty_point, 
                  float sparsification_tolerance,
                  int num_threads,
                  bool apply_sparsification = true) const;
    void fill_up_default_coords();
    std::string to_string() const;
};


#endif /* graph_nodes_hpp */
