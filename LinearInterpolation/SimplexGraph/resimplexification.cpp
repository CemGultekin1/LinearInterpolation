//
//  resimplexification_scoring.cpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 4/10/24.
//

#include "resimplexification.hpp"
#include <assert.h>
#include <iostream>

MomentCollector::MomentCollector(size_t degree,const MidpointTable& _midpoint_table,int nhtreads){
    moments = std::vector<std::vector<float>>{degree};
    midpoint_table = &_midpoint_table;
    counter = 0;
    num_threads = nhtreads;
};

void MomentCollector::add(long node){
    for(int d = 0; d < moments.size(); ++d){
        midpoint_table->increment_by_weights(moments[d], node, 1., d+1,num_threads);
    }
    ++counter;
}

void MomentCollector::remove(long node){
    for(int d = 0; d < moments.size(); ++d){
        midpoint_table->increment_by_weights(moments[d], node, -1., d+1,num_threads);
    }
    --counter;
}

float MomentCollector::variance() const{
    float _variance = 0.;
    auto mom1 = &moments[0];
    auto mom2 = &moments[1];
    float count = counter;
    for(int i = 0; i < mom1->size(); ++i ){
        _variance += mom2->at(i)/count - std::pow(mom1->at(i)/count,2);
    }
    _variance /= count;
    return _variance;
}

bool convexity_check(const PointWithDictionary& p,const long& n){
    for(auto &  [node,index]: p.node2index){
        if(n != node && p.weights.at(index) < 1e-5){
            return false;
        }
    }
    return true;
}


struct BiSimplex{
    PointWithDictionary p1;
    PointWithDictionary p2;
    int num_threads;
    TreeDescend path1;
    TreeDescend path2;
    BiSimplex(const MidpointTable& midpoint_table,const FacingNodes& fn,int);
    void descend();
};
BiSimplex::BiSimplex(const MidpointTable& midpoint_table,
                     const FacingNodes& fn,int _num_threads)
                    :p1(midpoint_table.max_dim),
                    p2(midpoint_table.max_dim ),
                    path1(p1,midpoint_table),
                    path2(p2,midpoint_table)
                    {
                        num_threads = _num_threads;
                        midpoint_table.increment_by_weights(p1.weights, fn.node1, 1.,1,num_threads);
                        midpoint_table.increment_by_weights(p2.weights, fn.node2, 1.,1,num_threads);
                        std::copy(fn.path1.begin(),fn.path1.end(),std::back_inserter(path1.descend_path));
                        std::copy(fn.path2.begin(),fn.path2.end(),std::back_inserter(path2.descend_path));
}

void BiSimplex::descend(){
    path2.follow_midpoint_with_step_discovery(&p1, num_threads);
    path1.follow_midpoint_with_step_discovery(&p2, num_threads);
}

void l2_variation(const MidpointTable& midpoint_table,FacingNodes& fn,int nthreads){
    BiSimplex bis(midpoint_table, fn,nthreads);
    log_sink.start_event("l2_variation/bis.descend()");
    bis.descend();
    log_sink.finish_event("l2_variation/bis.descend()");
    auto p1 = std::move(bis.p1);
    auto p2 = std::move(bis.p2);
    auto path1 = std::move(bis.path1);
    auto path2 = std::move(bis.path2);
    
    assert(p2.node2index.find(fn.node1) != p2.node2index.end());
    assert(p1.node2index.find(fn.node2) != p1.node2index.end());
    
    if(!convexity_check(p1,fn.node2)){
        return;
    }
    if(!convexity_check(p2,fn.node1)){
        return;
    }
    fn.convex_flag = true;
//    return;
    auto det1 = path1.determinant();
    auto det2 = path2.determinant();
    assert(det1 >= 0 && det2 >= 0);
    
    
    log_sink.start_event("l2_variation/MomentCollector");
    MomentCollector mc1{2,midpoint_table,nthreads};
    for(auto& p: p1.node2index){
        mc1.add(p.first);
    }
    
    float prior_penalty{mc1.variance()*det1};
    mc1.remove(fn.node1);
    mc1.add(fn.node2);
    prior_penalty += mc1.variance()*det2;
    
    float posterior_penalty{0.};
    for(auto node_index: p2.node2index){
        auto weight =p2.weights.at(node_index.second);
        if(weight < 0){
            continue;
        }
        auto det = det1*weight;
        mc1.remove(node_index.first);
        mc1.add(fn.node2);
        posterior_penalty += mc1.variance()*det;
        mc1.remove(fn.node2);
        mc1.add(node_index.first);
    }
    
    fn.prior_resim_cost = prior_penalty;
    fn.posterior_resim_cost =  posterior_penalty;
    log_sink.finish_event("l2_variation/MomentCollector");
}


void resimplexify(MidpointTable&  midpoint_table,const FacingNodes& facing_nodes,
                  std::vector<FacingNodes>& fnheap,const facing_node_comparison& compr,int nthreads){
    
    BiSimplex bis(midpoint_table, facing_nodes,nthreads);
    bis.descend();
    auto p1 = std::move(bis.p1);
    auto p2 = std::move(bis.p2);
    

    SparsePoint sp1{midpoint_table.max_dim};
    p1.to_point(sp1, -1., 0,false);
    
    SparsePoint sp2{midpoint_table.max_dim};
    p2.to_point(sp2, -1., 0,false);
    
    
    
    const auto alias1 = midpoint_table.add_midpoint(sp1, p1.weights.size(), facing_nodes.node1);
    const auto alias2 = midpoint_table.add_midpoint(sp2, p2.weights.size(), facing_nodes.node2);
    midpoint_table.insert_midpoint_on_hash(bis.path2.descend_path.hash(), alias1);
    midpoint_table.insert_midpoint_on_hash(bis.path1.descend_path.hash(), alias2);
    
    midpoint_table.update_depth(bis.path1.descend_path);
    midpoint_table.update_depth(bis.path2.descend_path);
    
    midpoint_table.facing_nodes.insert({alias1,alias2});
    midpoint_table.facing_nodes.insert({alias2,alias1});
    midpoint_table.facing_node_paths.insert({alias1,facing_nodes.path2});
    midpoint_table.facing_node_paths.insert({alias2,facing_nodes.path1});
    
    FacingNodes::HashFunction hash_fun{};
    
    for(int i = 0; i < 2; ++i){
        std::vector<FacingNodes> facing_nodes_list{};
        HashableGraphPath const* path;
        long node;
        size_t alias;
        if(i==0){
            path = &facing_nodes.path1;
            alias = alias2;
            node = facing_nodes.node1;
        }else{
            path = &facing_nodes.path2;
            alias = alias1;
            node = facing_nodes.node2;
        }
        Simplex sm{path, midpoint_table.max_dim};
        
        log_sink.start_event("FacingNodeFinder");
        FacingNodeFinder ffna{midpoint_table,sm,alias};
        ffna.remaining_nodes.erase(node);
        ffna.discover_all(facing_nodes_list);
        log_sink.finish_event("FacingNodeFinder");
        
        std::unordered_set<size_t> uset;
        
        for(auto &fn: facing_nodes_list){
            auto hash_value = hash_fun(fn);
            if(uset.find(hash_value) != uset.end()){
                continue;
            }
            facing_node_consistency(fn,midpoint_table);  
            log_sink.start_event("l2_variation");
            l2_variation(midpoint_table, fn,nthreads);
            log_sink.finish_event("l2_variation");
            if(!fn.convex_flag){
                continue;
            }
            
            uset.insert(hash_value);
            if(fn.resim_cost() > 0){
                continue;
            }
            fnheap.push_back(fn);
            std::push_heap(fnheap.begin(), fnheap.end(), compr);
        }
    }
}
