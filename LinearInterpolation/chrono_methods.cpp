//
//  chrono_methods.cpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 4/14/24.
//

#include "chrono_methods.hpp"


EventLog::EventLog(){
    t0 = std::chrono::system_clock::now();
    nano_dur = 0;
}
void EventLog::finish(){
    auto t1 = std::chrono::system_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);
    nano_dur = dur.count();
}

void LogSink::start_event(std::string st){
    logs[st] = EventLog{};
}

void LogSink::finish_event(std::string st){
    auto find = logs.find(st);
    if(find == logs.end()){
        return;
    }
    find->second.finish();
    auto dur = find->second.nano_dur;
    logs.erase(st);
    if(durs.find(st) == durs.end()){
        durs[st] = std::vector<long long>{};
    }
    durs[st].push_back(dur);
}

std::unordered_map<std::string,float> LogSink::average_times() const{
    std::unordered_map<std::string,float> cols{};
    for(const auto& [key,vec]: durs){
        long long sum = 0;
        for(const auto& val: vec){
            sum += val;
        }
        cols[key] = static_cast<float>(sum)/static_cast<float>(vec.size());
    }
    return cols;
}


std::unordered_map<std::string,long long> LogSink::sum_times() const{
    std::unordered_map<std::string,long long> cols{};
    for(const auto& [key,vec]: durs){
        long long sum = 0;
        for(const auto& val: vec){
            sum += val;
        }
        cols[key] = sum;
    }
    return cols;
}

void LogSink::report() const{
    for(auto [key,total_time] : sum_times()){
        auto milli_secs = static_cast<double>(total_time)/1e6;
        std::cout << key << " total ms = " << milli_secs << "\n";
    }
}



LogSink log_sink = LogSink{};
