//
//  debug_control.h
//  LinearInterpolation
//
//  Created by Cem Gultekin on 7/8/24.
//

#ifndef debug_control_h
#define debug_control_h

#define INTRPL_MTL_DEBUG

#ifdef INTRPL_MTL_DEBUG
//    #define NUMERICAL_CPU_CHECK
    #define DEBUG_PRINT(fmt, ...) \
        fprintf(stderr, "%s:%d:%s():\n\t " fmt, \
                strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__, \
                __LINE__, __func__, ##__VA_ARGS__)
#else
    #define DEBUG_PRINT(fmt, ...) ((void)0)
#endif

#endif /* debug_control_h */
