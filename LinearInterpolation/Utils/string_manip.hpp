//
//  string_manip.hpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 7/8/24.
//

#ifndef string_manip_hpp
#define string_manip_hpp

#include <string>
#include <cmath>

std::string toScientificNotation(float value, int significantFigures);


#define SCI_FL_STR(x) toScientificNotation(x,4).c_str()

#endif /* string_manip_hpp */
