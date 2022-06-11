#ifndef CUDA_CANNY
#define CUDA_CANNY

#include <cstdio>
#include <cstdint>

enum {
    NO_EDGE = 0,
    WEAK_EDGE = 128,
    STRONG_EDGE = 255,
};

/// Device hysteresis iteration flag
extern __device__ bool device_hysteresis_changes;

/// \brief Wrapper macro for CUDA error checking
///
/// Usage: CHECK_CUDA_ERRS( cudaMalloc(...) );
#define CHECK_CUDA_ERRS(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/// \brief Calculates offset for aligned shared memory access
///
/// Helper function that calculates the offset needed for aligned dynamic shared memory.
/// In practice calculates: pow(2, ceil(log2(size)+1))
__host__ __device__
inline int CalculateAlignedOffset(int size) {
    int offset = 2;
    while (size >>= 1) {
        offset <<= 1;
    }
    return offset;
}

__global__
void Pad2DGPU(const uint8_t* src, float* dst, int width, int height, int channels, int pad);

__global__
void CentralCropImageGPU(const uint8_t* src, uint8_t* dst, int width, int height, int channels, int pad);

__global__
void RGBToGrayscale(const uint8_t* src, uint8_t* dst, int width, int height, int channels);

__global__
void ApplyConv2D__basic(const float* src, float* dst, int width, int height, const float* kernel, int kernel_width, int kernel_height);

__global__
void ApplyConv2D__shared(const float* src, float* dst, int width, int height, const float* kernel, int kernel_width, int kernel_height);

/// Apply separable convolution on columns (vertically)
__global__
void ApplySeparableConv2DCols__basic(const float* src, float* dst, int width, int height, const float* kernel, int kernel_size);

/// Apply separable convolution on rows (horizontally)
__global__
void ApplySeparableConv2DRows__basic(const float* src, float* dst, int width, int height, const float* kernel, int kernel_size);

/// Apply separable convolution on columns (vertically)
__global__
void ApplySeparableConv2DCols__shared(const float* src, float* dst, int width, int height, const float* kernel, int kernel_size);

/// Apply separable convolution on rows (horizontally)
__global__
void ApplySeparableConv2DRows__shared(const float* src, float* dst, int width, int height, const float* kernel, int kernel_size);

__global__
void MagnitudeAndDirection(const float* horizontal, const float* vertical, float* mag, uint8_t* dir, int width, int height);

__global__
void NonMaxSuppression__basic(const float* mag, const uint8_t* dir, float* dst, int width, int height);

__global__
void NonMaxSuppression__nobranch(const float* mag, const uint8_t* dir, float* dst, int width, int height);

__global__
void Thresholding__basic(const float* src, uint8_t* dst, int width, int height, float high_thresh, float low_thresh);

__global__
void Thresholding__nobranch(const float* src, uint8_t* dst, int width, int height, float high_thresh, float low_thresh);

__global__
void Hysteresis__basic(uint8_t* src, int width, int height, int pad);

__global__
void Cleanup(uint8_t* src, int width, int height);



#endif //CUDA_CANNY