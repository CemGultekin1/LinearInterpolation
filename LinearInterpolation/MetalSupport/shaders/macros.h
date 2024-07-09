//
//  macros.h
//  LinearInterpolation
//
//  Created by Cem Gultekin on 7/6/24.
//

#ifndef macros_h
#define macros_h

#define xSTRINGIFY(s) STRINGIFY(s)
#define STRINGIFY(s) #s

#define quotation(x) STRINGIFY(x)


#define TEMPLATE_NAME(KRNL) KRNL ## _template


#define HST(KRNL,I)   STRINGIFY(KRNL ## _pt_ ## I)

#define HST_DCL(KRNL,ARGS,I)  template [[host_name(HST(KRNL,I))]] kernel void TEMPLATE_NAME(KRNL)<I>(ARGS)





#define KRNL_HDR(KRNL,ARGS) \
    template<ushort PER_THREAD> kernel void TEMPLATE_NAME(KRNL)(ARGS)

#define KRNL_TEMP(KERNEL_NAME,I,ARGS) \
    kernel void TEMPLATE_NAME<I>(KERNEL_NAME)(ARGS)


#define FIRST_LAYER_KRNL_NM first_layer
#define FIRST_LAYER_PSTR_KRNL_NM first_layer_posterior
#define FIND_EXIT_KRNL_NM find_exit_node
#define MDPNT_OPR_KRNL_NM midpoint_operation



#endif /* macros_h */
