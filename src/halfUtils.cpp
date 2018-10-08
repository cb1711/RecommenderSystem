#include "halfUtils.h"
#include <immintrin.h>


float half2float(uint16_t half) {
    return _cvtsh_ss(half);
}

uint16_t float2half(float f) {
    return _cvtss_sh(f,0);
}

#pragma intel optimization_parameter target_arch=CORE-AVX-I
void float2halfv(float** floats, uint16_t** halfs, int x, int y) {
    for(int j = 0; j < x; j++)
        for(int i = 0; i < y; i+= 4) {
            __m128 float_vector = _mm_load_ps(floats[j] + i);
            __m128i half_vector = _mm_cvtps_ph(float_vector, 0);
            _mm_store_si128((__m128i *)halfs[j] + i, half_vector);
        }
}

#pragma intel optimization_parameter target_arch=CORE-AVX-I
void half2floatv(float** floats, uint16_t** halfs, int x, int y) {
    for(int j=0; j < x; j++)
        for (int i = 0; i < y; i++) {
            __m128i half_vector = _mm_load_si128((__m128i *) halfs[j] + i);
            __m128 float_vector = _mm_cvtph_ps(half_vector);
            _mm_storeu_ps(floats[j] + i, float_vector);
        }
}