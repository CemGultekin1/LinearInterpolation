//
//  Header.h
//  LinearInterpolation
//
//  Created by Cem Gultekin on 7/8/24.
//

#ifndef formulas_h
#define formulas_h


inline float kappa_fun(float _mean, float dim ){
    return dim*(1 - _mean*(dim+1)/dim);
}


inline float beta_fun(float _mean_hat, float dim ,float kappa){
    return (1-_mean_hat)*kappa/dim/(kappa + 1);
}

inline float last_entry_fun(float _mean_hat,float kappa){
    return (1-_mean_hat)/(1+kappa);
}


inline float division_zero_permissible(float x,float y){
    return (y != 0) ? x/y : MAXFLOAT;
}
#endif /* Header_h */
