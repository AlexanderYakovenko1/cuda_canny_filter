#include <cstdio>

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"

#include "utils.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void SaveImage(const uint8_t* src, const char* path, int width, int height, int channels) {
    int comp = STBI_rgb;
    if (channels == 1) {
        comp = STBI_grey;
    }
    stbi_write_png(path, width, height, comp, src, width * channels);
}

__global__
void helloCUDA(float f)
{
    printf("Hello thread %d, f=%f\n", threadIdx.x, f);
}

__global__
void RGBToGrayscale(const uint8_t* src, uint8_t* dst, int width, int height, int channels) {
    int src_idx = (blockIdx.x * blockDim.x + threadIdx.x) * channels;
    int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (src_idx < width * height * channels) {
        dst[dst_idx] =  static_cast<uint8_t>(min(
            0.299f * src[src_idx    ] +
            0.587f * src[src_idx + 1] +
            0.114f * src[src_idx + 2],
            255.f
        ));
    }
}



// todo: unroll using templates
//__global__
//void ApplyConv2D(const float* src, float* dst, int width, int height, const float* kernel, int kernel_width, int kernel_height) {
//    int kernel_radius_width = kernel_width / 2;
//    int kernel_radius_height = kernel_height / 2;
//
//    // size of the area to load into shared mem
//    extern __shared__ float data[];
//    int processed_area_size = (kernel_height + blockDim.x) * (kernel_width + blockDim.y);
//
//    int top_left_x = blockIdx.x * blockDim.x;
//    int top_left_y = blockIdx.y * blockDim.y;
//    int first_idx = top_left_x * width + top_left_y;
//    // load image patch with apron
//    // todo: optimize for coalesced loading and use pitched array
//
//
//
//    for (int i = 0; i < height; ++i) {
//        for (int j = 0; j < width; ++j) {
//
//            float conv_val = 0.;
//            size_t idx, k_idx;
//            for (int k_i = -kernel_radius_height; k_i <= kernel_radius_height; ++k_i) {
//                for (int k_j = -kernel_radius_width; k_j <= kernel_radius_width; ++k_j) {
//                    if (i + k_i >= 0 && i + k_i < height && j + k_j >= 0 && j + k_j < width) {
//                        idx = (i + k_i) * width + j + k_j;
//                        k_idx = (k_i + kernel_radius_height) * kernel_width + k_j + kernel_radius_width;
//
//                        conv_val += src[idx] * kernel[k_idx];
//                    }
//                }
//            }
//
//            idx = i * width + j;
//            dst[idx] = conv_val;
//        }
//    }
//}

__global__
void ApplyConv2D__basic(const float* src, float* dst, int width, int height, const float* kernel, int kernel_width, int kernel_height) {
    int kernel_radius_width = kernel_width / 2;
    int kernel_radius_height = kernel_height / 2;

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < height && j < width) {
        float conv_val = 0.;
        size_t idx, k_idx;
        for (int k_i = -kernel_radius_height; k_i <= kernel_radius_height; ++k_i) {
            for (int k_j = -kernel_radius_width; k_j <= kernel_radius_width; ++k_j) {
                if (i + k_i >= 0 && i + k_i < height && j + k_j >= 0 && j + k_j < width) {
                    idx = (i + k_i) * width + j + k_j;
                    k_idx = (k_i + kernel_radius_height) * kernel_width + k_j + kernel_radius_width;

                    conv_val += src[idx] * kernel[k_idx];
                }
            }
        }

        idx = i * width + j;
        dst[idx] = conv_val;
    }
}

/// Calculates offset for aligned shared memory access
/// In practice calculates: pow(2, ceil(log2(size)+1))
__host__ __device__
int CalculateAlignedOffset(int size) {
    int offset = 2;
    while (size >>= 1) {
        offset <<= 1;
    }
    return offset;
}

__global__
void ApplyConv2D__shared(const float* src, float* dst, int width, int height, const float* kernel, int kernel_width, int kernel_height) {
    int kernel_radius_width = kernel_width / 2;
    int kernel_radius_height = kernel_height / 2;

    extern __shared__ float data[];
    float* shared_kernel = data;
    float* shared_src = &data[CalculateAlignedOffset(kernel_height * kernel_width)];

    int threadId = blockDim.y * threadIdx.y + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y;

    // load kernel to shared memory
    for (int i = threadId; i < kernel_height * kernel_width; i += blockSize) {
        shared_kernel[i] = kernel[i];
    }

    // load src image pixels into shared memory with apron
    int top_left_x = blockIdx.x * blockDim.x - kernel_radius_width;
    int top_left_y = blockIdx.y * blockDim.y - kernel_radius_height;
    int bottom_right_x = (blockIdx.x + 1) * blockDim.x + kernel_radius_width;
    int bottom_right_y = (blockIdx.y + 1) * blockDim.y + kernel_radius_height;


    for (int i = top_left_y + threadIdx.y; i < height && i < bottom_right_y; i += blockDim.y) {
        for (int j = top_left_x + threadIdx.x; j < width && j < bottom_right_x; j += blockDim.z) {
            if (i < 0 || j < 0) {
                continue;
            }

            int src_idx = i * width + j;
            int shared_src_idx = (i - top_left_y) * (blockDim.x + 2 * kernel_radius_width) + (j - top_left_x);


            shared_src[shared_src_idx] = src[src_idx];
        }
    }

    __syncthreads();

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int shared_i = threadIdx.y + kernel_radius_height;
    int shared_j = threadIdx.x + kernel_radius_width;

    if (i < height && j < width) {
        float conv_val = 0.;
        size_t idx, k_idx;
        for (int k_i = -kernel_radius_height; k_i <= kernel_radius_height; ++k_i) {
            for (int k_j = -kernel_radius_width; k_j <= kernel_radius_width; ++k_j) {

                // still needed since we are always inside shared memory but might be outside the source data
                if (i + k_i >= 0 && i + k_i < height && j + k_j >= 0 && j + k_j < width) {
                    idx = (shared_i + k_i) * (blockDim.x + 2 * kernel_radius_width) + shared_j + k_j;
                    k_idx = (k_i + kernel_radius_height) * kernel_width + k_j + kernel_radius_width;

                    conv_val += shared_src[idx] * shared_kernel[k_idx];
                }
            }
        }

        idx = i * width + j;
        dst[idx] = conv_val;
    }
}

/// Apply separable convolution on columns (vertically)
__global__
void ApplySeparableConv2DCols__basic(const float* src, float* dst, int width, int height, const float* kernel, int kernel_size) {
    int kernel_radius = kernel_size / 2;

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < height && j < width) {
        float conv_val = 0.;
        size_t idx, k_idx;
        for (int k_i = -kernel_radius; k_i <= kernel_radius; ++k_i) {
            if (i + k_i >= 0 && i + k_i < height) {
                idx = (i + k_i) * width + j;
                k_idx = k_i + kernel_radius;

                conv_val += src[idx] * kernel[k_idx];
            }
        }
        idx = i * width + j;
        dst[idx] = conv_val;
    }
}

/// Apply separable convolution on rows (horizontally)
__global__
void ApplySeparableConv2DRows__basic(const float* src, float* dst, int width, int height, const float* kernel, int kernel_size) {
    int kernel_radius = kernel_size / 2;

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < height && j < width) {
        float conv_val = 0.;
        size_t idx, k_idx;
        for (int k_j = -kernel_radius; k_j <= kernel_radius; ++k_j) {
            if (j + k_j >= 0 && j + k_j < width) {
                idx = i * width + j + k_j;
                k_idx = k_j + kernel_radius;

                conv_val += src[idx] * kernel[k_idx];
            }
        }
        idx = i * width + j;
        dst[idx] = conv_val;
    }
}

/// Apply separable convolution on columns (vertically)
__global__
void ApplySeparableConv2DCols__shared(const float* src, float* dst, int width, int height, const float* kernel, int kernel_size) {
    int kernel_radius = kernel_size / 2;

    extern __shared__ float data[];
    float* shared_kernel = data;
    float* shared_src = &data[CalculateAlignedOffset(kernel_size)];

    int threadId = blockDim.y * threadIdx.y + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y;

    // load kernel to shared memory
    for (int i = threadId; i < kernel_size; i += blockSize) {
        shared_kernel[i] = kernel[i];
    }

    // load src image pixels into shared memory with apron
    int top_left_x = blockIdx.x * blockDim.x;
    int top_left_y = blockIdx.y * blockDim.y - kernel_radius;
    int bottom_right_x = (blockIdx.x + 1) * blockDim.x;
    int bottom_right_y = (blockIdx.y + 1) * blockDim.y + kernel_radius;



    for (int i = top_left_y + threadIdx.y; i < height && i < bottom_right_y; i += blockDim.y) {
        for (int j = top_left_x + threadIdx.x; j < width && j < bottom_right_x; j += blockDim.x) {
            if (i < 0 || j < 0) {
                continue;
            }

            int src_idx = i * width + j;
            int shared_src_idx = (i - top_left_y) * blockDim.x + (j - top_left_x);

            shared_src[shared_src_idx] = src[src_idx];
        }
    }

    __syncthreads();

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int shared_i = threadIdx.y + kernel_radius;
    int shared_j = threadIdx.x;

    if (i < height && j < width) {
        float conv_val = 0.;
        size_t idx, k_idx;
        for (int k_i = -kernel_radius; k_i <= kernel_radius; ++k_i) {
            if (i + k_i >= 0 && i + k_i < height) {
                idx = (shared_i + k_i) * blockDim.x + shared_j;
                k_idx = k_i + kernel_radius;

                conv_val += shared_src[idx] * shared_kernel[k_idx];
            }
        }
        idx = i * width + j;
        dst[idx] = conv_val;
    }
}

/// Apply separable convolution on rows (horizontally)
__global__
void ApplySeparableConv2DRows__shared(const float* src, float* dst, int width, int height, const float* kernel, int kernel_size) {
    int kernel_radius = kernel_size / 2;

    extern __shared__ float data[];
    float* shared_kernel = data;
    float* shared_src = &data[CalculateAlignedOffset(kernel_size)];

    int threadId = blockDim.y * threadIdx.y + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y;

    // load kernel to shared memory
    for (int i = threadId; i < kernel_size; i += blockSize) {
        shared_kernel[i] = kernel[i];
    }

    // load src image pixels into shared memory with apron
    int top_left_x = blockIdx.x * blockDim.x - kernel_radius;
    int top_left_y = blockIdx.y * blockDim.y;
    int bottom_right_x = (blockIdx.x + 1) * blockDim.x + kernel_radius;
    int bottom_right_y = (blockIdx.y + 1) * blockDim.y;


    for (int i = top_left_y + threadIdx.y; i < height && i < bottom_right_y; i += blockDim.y) {
        for (int j = top_left_x + threadIdx.x; j < width && j < bottom_right_x; j += blockDim.x) {
            if (i < 0 || j < 0) {
                continue;
            }

            int src_idx = i * width + j;
            int shared_src_idx = (i - top_left_y) * (blockDim.x + 2 * kernel_radius) + (j - top_left_x);

            shared_src[shared_src_idx] = src[src_idx];
        }
    }

    __syncthreads();

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int shared_i = threadIdx.y;
    int shared_j = threadIdx.x + kernel_radius;

    if (i < height && j < width) {
        float conv_val = 0.;
        size_t idx, k_idx;
        for (int k_j = -kernel_radius; k_j <= kernel_radius; ++k_j) {
            if (j + k_j >= 0 && j + k_j < width) {
                idx = shared_i * (blockDim.x + 2 * kernel_radius) + shared_j + k_j;
                k_idx = k_j + kernel_radius;

                conv_val += shared_src[idx] * shared_kernel[k_idx];
            }
        }
        idx = i * width + j;
        dst[idx] = conv_val;
    }
}

// todo: change to no warp diversion
__global__
void MagnitudeAndDirection(const float* horizontal, const float* vertical, float* mag, uint8_t* dir, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < width * height) {
        float hor = horizontal[i];
        float ver = vertical[i];
        float cur_value = sqrtf(hor * hor + ver * ver);
        mag[i] = cur_value;

        float cur_angle = atanf(ver / hor) + M_PI_2f32;
        if (cur_angle > 3.5f * M_PI_4f32) {
            cur_angle -= 3.5f * M_PI_4f32;
        }
        dir[i] = static_cast<uint8_t>(ceilf(4 * cur_angle * M_1_PIf32 - 0.5f));
    }
}

// todo: replace ifs with calculations to avoid warp diversion
// todo: try shared memory
__global__
void NonMaxSuppression(const float* mag, const uint8_t* dir, float* dst, int width, int height) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && i < height - 1 && j > 0 && j < width - 1) {
        int idx = i * width + j;

        float max_along_grad = 0.0;
        if (dir[idx] == 0) {
            max_along_grad = max(mag[(i - 1) * width + (j)], mag[(i + 1) * width + (j)]);
        } else if (dir[idx] == 1) {
            max_along_grad = max(mag[(i - 1) * width + (j + 1)], mag[(i + 1) * width + (j - 1)]);
        } else if (dir[idx] == 2) {
            max_along_grad = max(mag[(i) * width + (j - 1)], mag[(i) * width + (j + 1)]);
        } else if (dir[idx] == 3) {
            max_along_grad = max(mag[(i - 1) * width + (j - 1)], mag[(i + 1) * width + (j + 1)]);
        }

        if (mag[idx] > max_along_grad) {
            dst[idx] = mag[idx];
        } else {
            dst[idx] = 0;
        }
    }
}

enum {
    NO_EDGE = 0,
    WEAK_EDGE = 128,
    STRONG_EDGE = 255,
};

// todo: replace ifs with calculations to avoid warp diversion
__global__
void Thresholding(const float* src, uint8_t* dst, int width, int height, float high_thresh, float low_thresh) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < width * height) {
        if (src[i] > high_thresh) {
            dst[i] = STRONG_EDGE;
        } else if (src[i] > low_thresh) {
            dst[i] = WEAK_EDGE;
        } else {
            dst[i] = NO_EDGE;
        }
    }
}

__device__ bool device_hysteresis_changes = false;

// todo: add shmem
__global__
void Hysteresis(uint8_t* src, int width, int height, int pad) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * width + j;

    pad = max(pad, 1);
    int offsets[8] = {
            -width - 1,
            -width    ,
            -width + 1,
            -1,
            1,
            width - 1,
            width    ,
            width + 1
    };

    __shared__ bool changes, anything;
    changes = true;
    anything = false;
    __syncthreads();

    while (changes) {
        changes = false;
        __syncthreads();

        if (i >= pad && i < height - pad && j >= pad && j < width - pad) {
            if (src[idx] == WEAK_EDGE) {
                for (int k = 0; k < 8; ++k) {
                    if (src[idx + offsets[k]] == STRONG_EDGE) {
                        src[idx] = STRONG_EDGE;
                        changes = true;
                        anything = true;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && threadIdx.y == 0 && anything) {
        device_hysteresis_changes = true;
    }
//    for (int i = pad; i < height - pad; ++i) {
//        for (int j = pad; j < width - pad; ++j) {
//            size_t idx = i * width + j;
//            if (src[idx] == WEAK_EDGE) {
//                src[idx] = NO_EDGE;
//            }
//        }
//    }
}

__global__
void Cleanup(uint8_t* src, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < width * height) {
        if (src[i] != STRONG_EDGE) {
            src[i] = NO_EDGE;
        }
    }
}

constexpr int BLOCK_SIZE = 32;
constexpr int MAX_THREADS = 1024;

int main()
{
    int width, height, channels;

    float sigma = 3;
    int radius;
    float* gaussian = GaussianKernel2D(sigma, &radius);
    float* gaussian_sep = GaussianKernel1D(sigma, &radius);
    float* dog_sep_diff = GaussianDerivativeKernel1D(sigma, &radius, true);
    float* dog_sep_norm = GaussianDerivativeKernel1D(sigma, &radius, false);

    uint8_t* image = stbi_load("../gz_a.png", &width, &height, &channels, STBI_rgb);
    uint8_t* output = AllocateArray<uint8_t>(width, height);
    float* pad = AllocateArray<float>(width + 2 * radius, height + 2 * radius);
    uint8_t* padd = AllocateArray<uint8_t>(width + 2 * radius, height + 2 * radius);

    uint8_t* cuda_image = nullptr;
    uint8_t* cuda_gray = nullptr;
    uint8_t* cuda_uint_pad = nullptr;
    float* cuda_gaussian = nullptr;
    float* cuda_gaussian_sep = nullptr;
    float* cuda_dog_sep_diff = nullptr;
    float* cuda_dog_sep_norm = nullptr;
    float* cuda_pad = nullptr;
    float* cuda_pad_buf = nullptr;
    float* cuda_pad_out = nullptr;
    float* cuda_pad_out_one = nullptr;
    gpuErrchk( cudaMalloc(&cuda_image, width * height * channels * sizeof(uint8_t)) );
    gpuErrchk( cudaMalloc(&cuda_gray, width * height * sizeof(uint8_t)) );
    gpuErrchk( cudaMalloc(&cuda_uint_pad, (width + 2 * radius) * (height + 2 * radius) * sizeof(uint8_t)) );
    gpuErrchk( cudaMalloc(&cuda_gaussian, (2 * radius + 1) * (2 * radius + 1) * sizeof(float)) );
    gpuErrchk( cudaMalloc(&cuda_gaussian_sep, (2 * radius + 1) * sizeof(float)) );
    gpuErrchk( cudaMalloc(&cuda_dog_sep_diff, (2 * radius + 1) * sizeof(float)) );
    gpuErrchk( cudaMalloc(&cuda_dog_sep_norm, (2 * radius + 1) * sizeof(float)) );
    gpuErrchk( cudaMalloc(&cuda_pad, (width + 2 * radius) * (height + 2 * radius) * sizeof(float)) );
    gpuErrchk( cudaMalloc(&cuda_pad_buf, (width + 2 * radius) * (height + 2 * radius) * sizeof(float)) );
    gpuErrchk( cudaMalloc(&cuda_pad_out, (width + 2 * radius) * (height + 2 * radius) * sizeof(float)) );
    gpuErrchk( cudaMalloc(&cuda_pad_out_one, (width + 2 * radius) * (height + 2 * radius) * sizeof(float)) );
    gpuErrchk( cudaMemcpy(cuda_image, image, width * height * channels * sizeof(uint8_t), cudaMemcpyHostToDevice) );

    // pointwise operation
    RGBToGrayscale<<<(width * height) / MAX_THREADS + 1, MAX_THREADS>>>(cuda_image, cuda_gray, width, height, channels);
    gpuErrchk( cudaMemcpy(output, cuda_gray, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost) );

//    SaveImage(output, "../cuda_gray.png", width, height, 1);

    Pad2D<uint8_t, float>(output, pad, width, height, 1, radius);
//    FloatToUint(pad, padd, width + 2 * radius, height + 2 * radius, 1);
//    SaveImage(padd, "../cuda_pad.png", width + 2 * radius, height + 2 * radius, 1);


    gpuErrchk( cudaMemcpy(cuda_pad, pad, (width + 2 * radius) * (height + 2 * radius) * sizeof(float), cudaMemcpyHostToDevice) );
    dim3 numBlocks((width + 2 * radius) / BLOCK_SIZE + 1, (height + 2 * radius) / BLOCK_SIZE + 1);
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpuErrchk( cudaMemcpy(cuda_gaussian, gaussian, (2 * radius + 1) * (2 * radius + 1) * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(cuda_gaussian_sep, gaussian_sep, (2 * radius + 1) * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(cuda_dog_sep_diff, dog_sep_diff, (2 * radius + 1) * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(cuda_dog_sep_norm, dog_sep_norm, (2 * radius + 1) * sizeof(float), cudaMemcpyHostToDevice) );
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        // print the CUDA error message and exit
        printf("CUDA errsor: %s\n", cudaGetErrorString(error)); exit(-1); }

    int memsize = CalculateAlignedOffset((2 * radius + 1) * (2 * radius + 1)) + (BLOCK_SIZE + 2 * radius) * (BLOCK_SIZE + 2 * radius);
    int memsize_sep = CalculateAlignedOffset((2 * radius + 1)) + (BLOCK_SIZE + 2 * radius) * (BLOCK_SIZE + 2 * radius);
    printf("memsize: %d\n", memsize_sep * sizeof(float));
    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
//    ApplyConv2D__shared<<<numBlocks, threadsPerBlock, memsize * sizeof(float)>>>(cuda_pad, cuda_pad_out,
//                                                       width + 2 * radius, height + 2 * radius, cuda_gaussian, 2 * radius + 1, 2 * radius + 1);
//    ApplyConv2D__basic<<<numBlocks, threadsPerBlock>>>(cuda_pad, cuda_pad_out,
//                                                       width + 2 * radius, height + 2 * radius, cuda_gaussian, 2 * radius + 1, 2 * radius + 1);
//    ApplySeparableConv2DRows__basic<<<numBlocks, threadsPerBlock>>>(cuda_pad, cuda_pad_buf,
//                                                                width + 2 * radius, height + 2 * radius, cuda_gaussian_sep, 2 * radius + 1);
//    ApplySeparableConv2DCols__basic<<<numBlocks, threadsPerBlock>>>(cuda_pad_buf, cuda_pad_out,
//                                                                    width + 2 * radius, height + 2 * radius, cuda_gaussian_sep, 2 * radius + 1);
    ApplySeparableConv2DRows__shared<<<numBlocks, threadsPerBlock, memsize_sep * sizeof(float)>>>(cuda_pad, cuda_pad_buf,
                                                                                                  width + 2 * radius, height + 2 * radius, cuda_dog_sep_diff, 2 * radius + 1);
    ApplySeparableConv2DCols__shared<<<numBlocks, threadsPerBlock, memsize_sep * sizeof(float)>>>(cuda_pad_buf, cuda_pad_out,
                                                                    width + 2 * radius, height + 2 * radius, cuda_dog_sep_norm, 2 * radius + 1);
    ApplySeparableConv2DRows__shared<<<numBlocks, threadsPerBlock, memsize_sep * sizeof(float)>>>(cuda_pad, cuda_pad_buf,
                                                                                                  width + 2 * radius, height + 2 * radius, cuda_dog_sep_norm, 2 * radius + 1);
    ApplySeparableConv2DCols__shared<<<numBlocks, threadsPerBlock, memsize_sep * sizeof(float)>>>(cuda_pad_buf, cuda_pad_out_one,
                                                                                                  width + 2 * radius, height + 2 * radius, cuda_dog_sep_diff, 2 * radius + 1);

    MagnitudeAndDirection<<<((width + 2 * radius) * (height + 2 * radius)) / MAX_THREADS + 1, MAX_THREADS>>>(cuda_pad_out, cuda_pad_out_one,
                                                                                                             cuda_pad_buf, cuda_uint_pad, width + 2 * radius, height + 2 * radius);

    NonMaxSuppression<<<numBlocks, threadsPerBlock>>>(cuda_pad_buf, cuda_uint_pad, cuda_pad_out, width + 2 * radius, height + 2 * radius);
    Thresholding<<<((width + 2 * radius) * (height + 2 * radius)) / MAX_THREADS + 1, MAX_THREADS>>>(cuda_pad_out, cuda_uint_pad, width + 2 * radius, height + 2 * radius, 1, 10);

    bool host_hysteresis_changes = true;
    while (host_hysteresis_changes) {
        host_hysteresis_changes = false;
        cudaMemcpyToSymbol(device_hysteresis_changes, &host_hysteresis_changes, sizeof(host_hysteresis_changes));

        Hysteresis<<<numBlocks, threadsPerBlock>>>(cuda_uint_pad, width + 2 * radius, height + 2 * radius, radius);
        cudaMemcpyFromSymbol(&host_hysteresis_changes, device_hysteresis_changes, sizeof(host_hysteresis_changes));
        printf("pow\n");
    }

//    Cleanup<<<((width + 2 * radius) * (height + 2 * radius)) / MAX_THREADS + 1, MAX_THREADS>>>(cuda_uint_pad, width + 2 * radius, height + 2 * radius);

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if(error != cudaSuccess) {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error)); exit(-1); }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms] to complete" << std::endl;

    cudaMemcpy(pad, cuda_pad_buf, (width + 2 * radius) * (height + 2 * radius) * sizeof(float), cudaMemcpyDeviceToHost);
    FloatToUint(pad, padd, width + 2 * radius, height + 2 * radius, 1);
    SaveImage(padd, "../cuda_mag_debug.png", width + 2 * radius, height + 2 * radius, 1);
    cudaMemcpy(padd, cuda_uint_pad, (width + 2 * radius) * (height + 2 * radius) * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    SaveImage(padd, "../gz_canny.png", width + 2 * radius, height + 2 * radius, 1);

    helloCUDA<<<1, 5>>>(1.2345f);
    cudaDeviceSynchronize();

    cudaFree(cuda_image);
    free(image);
    return 0;
}