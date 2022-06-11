#ifndef OPENMP_CANNY
#define OPENMP_CANNY

#include <iostream>
#include <cassert>
#include <cmath>

#include <omp.h>

enum {
    NO_EDGE = 0,
    WEAK_EDGE = 128,
    STRONG_EDGE = 255,
};

/// Great Cringe
void RGBToGrayscale(const uint8_t* src, uint8_t* dst, int width, int height, int channels);

void ApplyConv2D(const float* src, float* dst, int width, int height, const float* kernel, int kernel_width, int kernel_height);

void ApplySeparableConv2D(const float* src, float* dst, float* buffer, int width, int height, const float* kernel_h, int kernel_width, const float* kernel_v, int kernel_height);

void MagnitudeAndDirection(const float* horizontal, const float* vertical, float* mag, uint8_t* dir, int width, int height);

void NonMaxSuppression(const float* mag, const uint8_t* dir, float* dst, int width, int height);

void Thresholding(const float* src, uint8_t* dst, int width, int height, float high_thresh, float low_thresh);

void Hysteresis(uint8_t* src, int width, int height, int pad=1);



#endif //OPENMP_CANNY