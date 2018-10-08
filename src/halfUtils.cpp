#include "halfUtils.h"
#include <immintrin.h>
#include <iostream>


float half2float(uint16_t half) {
    return _cvtsh_ss(half);
}

uint16_t float2half(float f) {
    return _cvtss_sh(f,0);
}

//#pragma intel optimization_parameter target_arch=CORE-AVX-I
void f2hvUtil(float* floats, uint16_t* halfs) {
    __m128 float_vector = _mm_load_ps(floats);
    __m128i half_vector = _mm_cvtps_ph(float_vector, 0);
    _mm_store_si128((__m128i *)halfs, half_vector);
}

void float2halfv(float* floats, uint16_t* halfs, int x) {
    for(int i = 0; i < x; i+= 4)
        f2hvUtil(floats+i, halfs+i);
}

//#pragma intel optimization_parameter target_arch=CORE-AVX-I
void h2fvUtil(float *floats, uint16_t* halfs){
    __m128i half_vector = _mm_load_si128((__m128i *) halfs);
    __m128 float_vector = _mm_cvtph_ps(half_vector);
    _mm_storeu_ps(floats, float_vector);
}

void half2floatv(float* floats, uint16_t* halfs, int x) {
    for (int i = 0; i < x; i+=4)
        h2fvUtil(floats+i,halfs+i);
}
