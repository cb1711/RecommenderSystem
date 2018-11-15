#ifndef RECOMMENDERSYSTEM_HALFUTILS_H
#define RECOMMENDERSYSTEM_HALFUTILS_H

#include <stdint.h>

float half2float(uint16_t half);

uint16_t float2half(float f);

void float2halfv(float* floats, uint16_t* halfs, int x);

void half2floatv(float* floats, uint16_t* halfs, int x);

#endif //RECOMMENDERSYSTEM_HALFUTILS_H
