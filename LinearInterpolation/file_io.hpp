//
//  file_io.hpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 4/14/24.
//

#ifndef file_io_hpp
#define file_io_hpp

#include <stdio.h>
#include <iostream>
#include <iomanip>

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "vector<" << typeid(T).name() << ">#" << v.size() << ":\n";
    for(const auto& p: v){
        os << p << " ";
    }
    return os;
}

//template <typename T>
//struct VectorIO{
//    
//};


template <typename T>
std::istream& operator>>(std::istream& os, std::vector<T>& v)
{
    for(size_t i = 0; i < v.capacity() ; ++i){
        os >> std::back_inserter(v);
    }
    return os;
}

#endif /* file_io_hpp */
