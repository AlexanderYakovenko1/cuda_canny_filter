#include "canny.cuh"

__device__ bool device_hysteresis_changes = false;

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

__global__
void Pad2DGPU(const uint8_t* src, float* dst, int width, int height, int channels, int pad) {
    int padded_width = (width + 2 * pad);
    int padded_height = (height + 2 * pad);

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < padded_height && j < padded_width) {
        for (int c = 0; c < channels; ++c) {
            int e_i = i - pad, e_j = j - pad;  // effective coordinates
            if (i < pad) {
                e_i = pad - i;
            } else if (i >= height + pad) {
                e_i = height - (i - height - pad) - 1;
            }
            if (j < pad) {
                e_j = pad - j;
            } else if (j >= width + pad) {
                e_j = width - (j - width - pad) - 1;
            }

            int src_idx = (e_i * width + e_j) * channels + c;
            int pad_idx = (i * padded_width + j) * channels + c;
            dst[pad_idx] = src[src_idx];
        }
    }
}

__global__
void CentralCropImageGPU(const uint8_t* src, uint8_t* dst, int width, int height, int channels, int pad) {
    int cropped_width = (width - 2 * pad);
    int cropped_height = (height - 2 * pad);

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int offset = (pad * width) * channels;
    if (i < cropped_height && j < cropped_width) {
        for (int c = 0; c < channels; ++c) {
            int src_idx = offset + (i * width + j + pad) * channels + c;
            int crop_idx = (i * cropped_width + j) * channels + c;
            dst[crop_idx] = src[src_idx];
        }
    }
}


// todo: unroll using templates
// todo: optimize for coalesced loading and use pitched array

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
void MagnitudeAndDirection__basic(const float* horizontal, const float* vertical, float* mag, uint8_t* dir, int width, int height) {
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
void NonMaxSuppression__basic(const float* mag, const uint8_t* dir, float* dst, int width, int height) {
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

// todo: replace ifs with calculations to avoid warp diversion
__global__
void Thresholding__basic(const float* src, uint8_t* dst, int width, int height, float high_thresh, float low_thresh) {
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

// todo: add shmem
__global__
void Hysteresis__basic(uint8_t* src, int width, int height, int pad) {
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
