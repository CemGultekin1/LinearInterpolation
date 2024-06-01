//
//  chrono_methods.hpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 4/14/24.
//

#ifndef chrono_methods_hpp
#define chrono_methods_hpp

#include <stdio.h>
#include <chrono>
#include <unordered_map>
#include <vector>
#include <iostream>

struct EventLog{
    std::chrono::time_point<std::chrono::system_clock> t0;
    long long nano_dur;
    EventLog();
    void finish();
    bool is_finished() const;
};

struct LogSink{
    std::unordered_map<std::string,EventLog> logs;
    std::unordered_map<std::string,std::vector<long long>> durs;
    void start_event(std::string);
    void finish_event(std::string);
    std::unordered_map<std::string,float> average_times() const;
    std::unordered_map<std::string,long long> sum_times() const;
    void report() const;
};

extern LogSink log_sink;
#endif /* chrono_methods_hpp */
