//
//  linearAlgebra.cpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 3/31/24.
//

#include "numerical_ops.hpp"
#include "chrono_methods.hpp"
// Function to find maximum element in a portion of the vector
void find_escape_node(
              const std::vector<float>& midpoint_weights,
              const std::vector<long>& midpoint_nodes,
              const std::vector<float>& x_weights,
              const std::unordered_map<long,unsigned long>& x_nodes_map,
              size_t start, size_t end, float & local_minimizer_weight,size_t & local_minimizer_index){
    float local_min = std::numeric_limits<float>::max();
    long local_minimizer = std::numeric_limits<long>::max();
    for (size_t i = start; i < end; ++i) {
        auto c = midpoint_nodes[i];
        auto xLoc = x_nodes_map.find(c);
        if( xLoc == x_nodes_map.end()){
            continue;
        }
        auto xw = x_weights[xLoc->second];
        auto mw = midpoint_weights[i];
        if(mw <= xw){
            continue;
        }
        mw = mw/(mw - xw);
        if(mw < local_min){
            local_min = mw;
            local_minimizer = i;
        }
    }
    local_minimizer_weight = local_min;
    local_minimizer_index = local_minimizer;
}


size_t parallel_find_escape_node(const std::vector<float>& midpoint_weights,
                           const std::vector<long>& midpoint_nodes,
                           const std::vector<float>& x_weights,
                           const std::unordered_map<long,unsigned long>& x_nodes_map,
                           int num_threads) {
    if(num_threads == 0){
        float minimizer_weights = 0;
        size_t minimizer_indexes = 0;
        find_escape_node(midpoint_weights,midpoint_nodes,x_weights,
                         x_nodes_map,0,static_cast<int>(midpoint_weights.size()),minimizer_weights,minimizer_indexes);
        return minimizer_indexes;
    }
    std::vector<std::thread> threads(num_threads);
    std::vector<float> minimizer_weights(num_threads);
    std::vector<size_t> minimizer_indexes(num_threads);
    
    size_t chunk_size = midpoint_weights.size() / num_threads;
    size_t start = 0;
    for (int i = 0; i < num_threads; ++i) {
        size_t end = (i == num_threads - 1) ? midpoint_weights.size() : start + chunk_size;
        threads[i] = std::thread(find_escape_node,
                                 std::cref(midpoint_weights),
                                 std::cref(midpoint_nodes),
                                 std::ref(x_weights),
                                 std::ref(x_nodes_map),
                                 start,
                                 end,
                                 std::ref(minimizer_weights[i]),
                                 std::ref(minimizer_indexes[i]));
        start = end;
    }
    for (int i = 0; i < num_threads; ++i) {
        threads[i].join();
    }
    float total_min{std::numeric_limits<float>::max()};
    size_t total_index_minimizer{std::numeric_limits<size_t>::max()};
    for(int i = 0; i < num_threads; ++i){
        auto index = minimizer_indexes[i];
        auto weight = minimizer_weights[i];
        if(weight < total_min){
            total_min = weight;
            total_index_minimizer = index;
        }
    }
    return total_index_minimizer;
}

void apply_midpoint_operation(const std::vector<float>& midpoint_weights,
                              const std::vector<long> & midpoint_nodes,
                              std::vector<float>& x_weights,
                              std::unordered_map<long,unsigned long>& x_nodes_map,
                              float step_size,size_t start, size_t end){
    for (size_t i = start; i < end; ++i) {
        auto c = midpoint_nodes.at(i);
        auto xLoc = x_nodes_map.find(c);
        if( xLoc == x_nodes_map.end()){
            continue;
        }
        auto mw = midpoint_weights.at(i);
        x_weights.at(xLoc->second) -= mw*step_size;
    }
}


void parallel_apply_midpoint_operation(
                    const std::vector<float>& midpoint_weights,
                    const std::vector<long>&  midpoint_nodes,
                    std::vector<float>& x_weights,
                    std::unordered_map<long,size_t>& x_nodes_map,
                    size_t escape_index,
                    long midpoint_node,
                    int num_threads){
    auto escape_node = midpoint_nodes.at(escape_index);
    auto beta = midpoint_weights.at(escape_index);
    auto escape_node_x_index = x_nodes_map.find(escape_node)->second;
    auto alpha = x_weights.at(escape_node_x_index);
    auto step_size = alpha/beta;
    
    if(num_threads == 0){
        apply_midpoint_operation(midpoint_weights,midpoint_nodes,
                                 x_weights,x_nodes_map,step_size,0,static_cast<int>(midpoint_weights.size()));
    }else{
        std::vector<std::thread> threads(num_threads);
        size_t chunk_size = midpoint_weights.size() / num_threads;
        size_t start = 0;
        for (int i = 0; i < num_threads; ++i) {
            size_t end = (i == num_threads - 1) ? midpoint_weights.size() : start + chunk_size;
            threads[i] = std::thread(apply_midpoint_operation,
                                     std::cref(midpoint_weights),
                                     std::cref(midpoint_nodes),
                                     std::ref(x_weights),
                                     std::ref(x_nodes_map),
                                     step_size,
                                     start,
                                     end);
            start = end;
        }
        for (int i = 0; i < num_threads; ++i) {
            threads[i].join();
        }
    }
    x_weights.at(escape_node_x_index) = step_size;
    x_nodes_map.erase(escape_node);
    x_nodes_map.insert({midpoint_node,escape_node_x_index});
    
}

void first_layer(std::vector<float>& x_weights,float& sum,size_t start,size_t end){
    float local_sum{0.};
    float d = static_cast<float>(x_weights.size());
    for(size_t i = start; i <end; ++i ){
        x_weights[i] /= d;
        local_sum += x_weights[i];
    }
    sum = local_sum;
}

void parallel_apply_first_layer(
                    std::vector<float>& x_weights,
                    int num_threads){
    /*
     y[i]  = x[i]/d
     y[-1] = 1-sum(x)/d
     
     x[i] = y[i]*d     
     */
    float total_sum = 1.;
    if(num_threads == 0){
        first_layer(x_weights,total_sum,0,static_cast<int>(x_weights.size()));
        total_sum = 1.-total_sum;
    }else{
        std::vector<std::thread> threads(num_threads);
        std::vector<float> sums(num_threads);
        size_t chunk_size = x_weights.size() / num_threads;
        size_t start = 0;
        for (int i = 0; i < num_threads; ++i) {
            size_t end = (i == num_threads - 1) ? x_weights.size() : start + chunk_size;
            threads[i] = std::thread(first_layer,
                                     std::ref(x_weights),
                                     std::ref(sums[i]),
                                     start,
                                     end);
            start = end;
        }
        for (int i = 0; i < num_threads; ++i) {
            threads[i].join();
        }
        
        for(int i = 0; i < num_threads; ++i){
            total_sum -= sums[i];
        }
    }
    
    x_weights.push_back(total_sum);
    
}



void find_max(const std::vector<float>& x_weights,float& max_val,size_t start,size_t end){
    float local_max{std::numeric_limits<float>::min()};
    for(size_t i = start; i < end; ++i){
        local_max = std::max(local_max,x_weights[i]);
    }
    max_val = local_max;
}

void parallel_sparsification(const std::vector<float>& x_weights,
                             const std::unordered_map<long,size_t>& x_nodes,
                             std::vector<float>& new_x_weights,
                             std::vector<long>& new_x_nodes,
                             float tolerance,
                             int num_threads){
    float lowlim{std::numeric_limits<float>::min()};
    if(num_threads == 0){
        find_max(x_weights,lowlim,0,static_cast<int>(x_weights.size()));
    }else{
        std::vector<std::thread> threads(num_threads);
        std::vector<float> maxs(num_threads);
        size_t chunk_size = x_weights.size() / num_threads;
        size_t start = 0;
        for (int i = 0; i < num_threads; ++i) {
            size_t end = (i == num_threads - 1) ? x_weights.size() : start + chunk_size;
            threads[i] = std::thread(find_max,
                                     std::ref(x_weights),
                                     std::ref(maxs[i]),
                                     start,
                                     end);
            start = end;
        }
        for (int i = 0; i < num_threads; ++i) {
            threads[i].join();
        }
        for(int i = 0; i < num_threads; ++i){
            lowlim = std::max(lowlim,maxs[i]);
        }
    }
    lowlim *= tolerance;
    size_t index = 0;
    for(const auto & p:x_nodes){
        auto w = x_weights[p.second];
        if(w < lowlim){
            continue;
        }
        new_x_weights.push_back(w);
        new_x_nodes.push_back(p.first);
        ++index;
    }
}



bool parallel_set_contains(const std::unordered_map<long,size_t>& x_nodes, const std::vector<long>& midpoint_nodes){
    for(const auto & p: midpoint_nodes){
        if(x_nodes.find(p) == x_nodes.end()){
            return false;
        }
    }
    return true;
}

bool parallel_set_contains(const std::unordered_set<long>& x_nodes, const std::vector<long>& midpoint_nodes){
    for(const auto & p: midpoint_nodes){
        if(x_nodes.find(p) == x_nodes.end()){
            return false;
        }
    }
    return true;
}

bool parallel_vector_contains(const long& node, const std::vector<long>& nodes){
    for(auto p:nodes){
        if(p == node){
            return true;
        }
    }
    return false;
}



void pointwise_addition(std::vector<float>& out,const std::vector<float>& inc,float weight,int power,int start,int end){
    if(power == 1){
        for(size_t i = start; i < end; ++i){
            out[i] += weight*inc[i];
        }
    }else{
        for(size_t i = start; i < end; ++i){
            out[i] += weight*std::pow(inc[i],power);
        }
    }
}

void parallel_pointwise_addition(std::vector<float>& out,const std::vector<float>& inc,float weight,int power,int num_threads){
    if(num_threads == 0){
        pointwise_addition(out,inc,weight,power,0,static_cast<int>(inc.size()));
        return;
    }
    log_sink.start_event("parallel_pointwise_addition");
    std::vector<std::thread> threads(num_threads);
    size_t chunk_size = inc.size() / num_threads;
    size_t start = 0;
    for (int i = 0; i < num_threads; ++i) {
        size_t end = (i == num_threads - 1) ? inc.size() : start + chunk_size;
        threads[i] = std::thread(pointwise_addition,
                                 std::ref(out),
                                 std::cref(inc),
                                 weight,
                                 power,
                                 start,
                                 end);
        start = end;
    }
    for (int i = 0; i < num_threads; ++i) {
        threads[i].join();
    }
    log_sink.finish_event("parallel_pointwise_addition");
}
