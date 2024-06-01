//
//  simplex_tree.cpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 4/2/24.
//

#include "simplex_tree.hpp"
#include "collisions.hpp"
#include <exception>
#include "facing_node.hpp"
#include <assert.h>
#include "resimplexification.hpp"
#include "chrono_methods.hpp"
SimplexTree::SimplexTree(int nthreads,float spt){
    nthreads = std::max(0,nthreads);
    max_nthreads = std::min(nthreads,static_cast<int>(std::thread::hardware_concurrency()));
    midpoint_table = MidpointTable();
    sparsification_tolerance = spt;
}

void SimplexTree::operator()(std::vector<float>& x,SparsePoint& sp,bool sparse) const{
    PointWithDictionary point_wd{x};
    TreeDescend td(point_wd,midpoint_table);
    td.descend(max_nthreads);
    if(sparse){
        point_wd.to_point(sp, sparsification_tolerance, max_nthreads);
    }else{
        point_wd.to_point(sp, 0., max_nthreads);
    }
    
}

void collision_check(const MidpointTable& midpoint_table,Simplex const* leaf_simplex){
    for(const auto& n : midpoint_table.midpoints.back().nodes){
        if(leaf_simplex->nodes.find(n) == leaf_simplex->nodes.end()){
            std::string error_message = "leaf simplex is inconsistent!\nmidpoint coordinates = ";
            auto nodes = midpoint_table.midpoints.back().nodes;
            std::sort(nodes.begin(), nodes.end());
            std::vector<long> leaf_simplex_nodes{};
            std::copy(leaf_simplex->nodes.begin(), leaf_simplex->nodes.end(), std::back_inserter(leaf_simplex_nodes));
            std::sort(leaf_simplex_nodes.begin(),leaf_simplex_nodes.end());
            for(int i = 0; i < nodes.size(); i ++ ){
                error_message.append(std::to_string(nodes[i])+ ",");
            }
            error_message.append("\nsimplex coordinates = ");
            for(int i = 0; i < leaf_simplex_nodes.size(); i ++ ){
                error_message.append(std::to_string(leaf_simplex_nodes[i])+ ",");
            }
            throw std::invalid_argument(error_message);
        }
    }
}
size_t SimplexTree::add_midpoint(const std::vector<float>& x){
    auto node = midpoint_table.add_node(x);    
    PointWithDictionary point_wd{x};
    TreeDescend td(point_wd,midpoint_table);
    td.descend(max_nthreads);
    SparsePoint sparse_point(midpoint_table.max_dim);
    point_wd.to_point(sparse_point, sparsification_tolerance, max_nthreads);
    if(sparse_point.nodes.size() == 1){
        return std::numeric_limits<size_t>::max();
    }
    auto alias = midpoint_table.add_midpoint(sparse_point, point_wd.weights.size(), node);
    midpoint_table.update_depth(td.descend_path);
    std::cout << "new alias = " << alias << "\n";
    
//    if(midpoint_table.midpoints.size() == 1){
    midpoint_table.insert_midpoint_on_hash(td.descend_path.hash(),alias);
//        return alias;
//    }
    
    
    
    Simplex leaf_simplex{&td.descend_path,sparse_point.nodes};
    Simplex collision_root_simplex{&td.descend_path,sparse_point.nodes};
    std::deque<MidpointStepMultiplicity> multiplicities{};
    simplex_tree_ascend(midpoint_table, leaf_simplex, collision_root_simplex, multiplicities);
    SimplexIterator sit{&leaf_simplex, &midpoint_table, &collision_root_simplex, &multiplicities, alias};
    std::vector<FacingNodes> facing_nodes_list{};    
    int num_simplexes = 0;
    while(sit.foward()){
        auto simplex = sit.leaf_simplex;
        midpoint_table.insert_midpoint_on_hash(simplex->descend_path.hash(),alias);
        collision_check(midpoint_table,simplex);
        midpoint_table.update_depth(simplex->descend_path);
        ++num_simplexes;
        
        log_sink.start_event("FacingNodeFinder");
        FacingNodeFinder ffna{midpoint_table,*simplex,alias};
        ffna.discover_all(facing_nodes_list);
        log_sink.finish_event("FacingNodeFinder");
    }
    FacingNodes::HashFunction hash_fun{};
    std::unordered_set<size_t> uset;
    for(auto &fn: facing_nodes_list){
        auto hash_value = hash_fun(fn);
        if(uset.find(hash_value) != uset.end()){
            continue;
        }
        log_sink.start_event("l2_variation");
        facing_node_consistency(fn,midpoint_table);
        l2_variation(midpoint_table, fn,max_nthreads);
        log_sink.finish_event("l2_variation");
        if(!fn.convex_flag){
            continue;
        }
        uset.insert(hash_value);
    }
    
    facing_node_comparison fncompr{false};
    std::make_heap(facing_nodes_list.begin(),facing_nodes_list.end(),fncompr);
    while(!facing_nodes_list.empty()){
        std::pop_heap(facing_nodes_list.begin(), facing_nodes_list.end(),fncompr);
        auto fn = facing_nodes_list.back();
        facing_nodes_list.pop_back();
        if(fn.resim_cost() > 0){
            facing_nodes_list.clear();            
            break;
        }
        std::cout << fn.to_string() << "\n";
        resimplexify(midpoint_table,fn,facing_nodes_list,fncompr,max_nthreads);
    }
    std::cout << "num_simplexes = " << num_simplexes << "\n";
    return alias;
}
