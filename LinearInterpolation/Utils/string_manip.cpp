//
//  string_manip.cpp
//  LinearInterpolation
//
//  Created by Cem Gultekin on 7/8/24.
//

#include "string_manip.hpp"


std::string toScientificNotation(float value, int significantFigures){
    if (value == 0.0) {
        return "0";
    }

    int exponent = std::floor(std::log10(std::abs(value)));
    float normalizedValue = value / std::pow(10.0, exponent);

    std::string result = std::to_string(normalizedValue);

    // Trim to significant figures
    result.erase(significantFigures + (result.find(".") != std::string::npos), std::string::npos); // Remove trailing 0s in decimal part
    
    // Append exponent
    result += "e" + std::to_string(exponent);

    return result;
}
