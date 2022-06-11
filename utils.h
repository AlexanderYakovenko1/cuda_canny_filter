#ifndef UTILS_H
#define UTILS_H

#include <cstdlib>
#include <iostream>
#include <cmath>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"

/// Wrapper function for saving images
/// @param[in] src                      Source 8-bit image
/// @param     path                     Path to save
/// @param     width, height, channels  Image dimensions
void SaveImage(const uint8_t* src, const char* path, int width, int height, int channels) {
    int comp = STBI_rgb;
    if (channels == 1) {
        comp = STBI_grey;
    }
    stbi_write_png(path, width, height, comp, src, width * channels);
}

template<typename T>
T* AllocateArray(int width, int height, int channels=1) {
    size_t image_size = width * height * channels;
    T* image = static_cast<T*>(calloc(image_size, sizeof(T)));
    return image;
}

void UintToFloat(const uint8_t* src, float* dst, int width, int height, int channels) {
    size_t total_len = width * height * channels;
    while (total_len--) {
        *dst++ = static_cast<float>(*src++);
    }
}

void FloatToUint(const float* src, uint8_t* dst, int width, int height, int channels, float mul=1.f, float add=0.f) {
    size_t total_len = width * height * channels;
    while (total_len--) {
        *dst++ = static_cast<uint8_t>(std::min(std::max(mul * *src++ + add, 0.f), 255.f));
    }
}

///
/// \brief Zero-pad an array by pad pixels.
///
/// No safety checks. All arrays have to be preallocated before passing into this function.
/// \tparam T1 Source array type
/// \tparam T2 Destination array type (note: implicit cast from T1 to T2)
/// \param[in] src Source array of shape (width, height, channels)
/// \param[out] dst Destination array of shape (width + 2 * pad, height + 2 * pad, channels)
/// \param[in] width, height, channels Dimensions of the source array
/// \param[in] pad The amount of padding, must non-negative
template<typename T1, typename T2>
void Pad2D(const T1* src, T2* dst, int width, int height, int channels, int pad) {
    int padded_width = (width + 2 * pad);
    int padded_height = (height + 2 * pad);

    for (int i = 0; i < padded_height; ++i) {
        for (int j = 0; j < padded_width; ++j) {
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
                if (e_i < 0 || e_j < 0 || e_i > height || e_j > width) {
                    printf("eff: %d %d\nact: %d %d\n\n", e_i, e_j, i, j);
                    exit(1);
                }
                size_t src_idx = (e_i * width + e_j) * channels + c;
                size_t pad_idx = (i * padded_width + j) * channels + c;
                dst[pad_idx] = src[src_idx];
            }
        }
    }
}

// perform central crop on an image by pad pixels
// no safety checks
void CentralCropImage(const uint8_t* src, uint8_t* dst, int width, int height, int channels, int pad) {
    int cropped_width = (width - 2 * pad);
    int cropped_height = (height - 2 * pad);

    size_t offset = (pad * width) * channels;
    for (int i = 0; i < cropped_height; ++i) {
        for (int j = 0; j < cropped_width; ++j) {
            for (int c = 0; c < channels; ++c) {
                size_t src_idx = offset + (i * width + j + pad) * channels + c;
                size_t crop_idx = (i * cropped_width + j) * channels + c;
                dst[crop_idx] = src[src_idx];
            }
        }
    }
}

// derivative of gaussian kernel calculation
// $$\frac{\partial{G(x,y)}}{\partial{x}}= -x \frac{1}{2 \pi \sigma^4} \exp{\frac{x^2+y^2}{2\sigma^2}}$$
float* GaussianDerivativeKernel1D(float sigma, int* radius, bool coord=true) {
    // cutoff radius via 3-sigma rule
    *radius = static_cast<int>(ceil(3. * sigma));
    float sigma_2 = sigma * sigma;
    float norm = 1. / (sqrt(2. * M_PI) * sigma_2);
    float* kernel = static_cast<float*>(calloc(2 * (*radius) + 1, sizeof(float)));
    for (int i = -(*radius); i <= (*radius); ++i) {
        if (coord) {
            kernel[i + (*radius)] = -i * norm * exp(-i * i / (2 * sigma_2));
        } else {
            kernel[i + (*radius)] = norm * exp(-i * i / (2 * sigma_2));
        }
    }
    return kernel;
}

float* GaussianDerivativeKernel2D(float sigma, int* radius, bool vertical=false) {
    // cutoff radius via 3-sigma rule
    *radius = static_cast<int>(ceil(3. * sigma));
    int diameter = 2 * (*radius) + 1;
    float sigma_2 = sigma * sigma;
    float sigma_4 = sigma_2 * sigma_2;
    float norm = 1. / (2. * M_PI * sigma_4);
    float* kernel = static_cast<float*>(calloc(diameter * diameter, sizeof(float)));
    for (int i = -(*radius); i <= (*radius); ++i) {
        for (int j = -(*radius); j <= (*radius); ++j) {
            size_t idx = (i + (*radius)) * diameter + j + (*radius);
            if (vertical) {
                kernel[idx] = -j * norm * exp(-(i * i + j * j) / (2 * sigma_2));
            } else {
                kernel[idx] = -i * norm * exp(-(i * i + j * j) / (2 * sigma_2));
            }
        }
    }
    return kernel;
}

float* GaussianKernel1D(float sigma, int* radius) {
    // cutoff radius via 3-sigma rule
    *radius = static_cast<int>(ceil(3. * sigma));
    float sigma_2 = sigma * sigma;
    float norm = 1. / (sqrt(2. * M_PI) * sigma);
    float* kernel = static_cast<float*>(calloc(2 * (*radius) + 1, sizeof(float)));
    for (int i = -(*radius); i <= (*radius); ++i) {
        kernel[i + (*radius)] = norm * exp(-i * i / (2 * sigma_2));
    }
    return kernel;
}

float* GaussianKernel2D(float sigma, int* radius) {
    // cutoff radius via 3-sigma rule
    *radius = static_cast<int>(ceil(3. * sigma));
    int diameter = 2 * (*radius) + 1;
    float sigma_2 = sigma * sigma;
    float norm = 1. / (2. * M_PI * sigma_2);
    float* kernel = static_cast<float*>(calloc(diameter * diameter, sizeof(float)));
    for (int i = -(*radius); i <= (*radius); ++i) {
        for (int j = -(*radius); j <= (*radius); ++j) {
            size_t idx = (i + (*radius)) * diameter + j + (*radius);
            kernel[idx] = norm * exp(-(i * i + j * j) / (2 * sigma_2));
        }
    }
    return kernel;
}

template<typename T>
void PrintArr(T* arr, int len) {
    for (int i = 0; i < len; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

void ParseArguments(int argc, char** argv, char** input, char** output, float* sigma, float* low_thresh, float* high_thresh, int* num_runs) {
    const char help_str[] = "Canny Filter Application "
                            "usage: %s sigma thr_high thr_low input_image output_image [num_runs]\n\n"
                            "Filter options:\n"
                            "  sigma\t\tGaussian filter sigma value, must be a positive float\n"
                            "  thr_high\tHigh threshold of Canny filter, must be a positive float\n"
                            "  thr_low\tLow threshold of Canny filter\n\n"
                            "  num_runs\t(optional) number of runs for benchmarking"
                            "Help:\n"
                            "  -h\t\tShow this text\n";
    if (argc < 6 || (argc > 1 && argv[1] == "-h")) {
        printf(help_str, argv[0]);
        exit(0);
    }
    *sigma = strtof(argv[1], nullptr);
    *low_thresh = strtof(argv[2], nullptr);
    *high_thresh = strtof(argv[3], nullptr);
    *input = argv[4];
    *output = argv[5];
    if (*sigma < 0 || *low_thresh < 0 || *high_thresh < 0) {
        printf("Expected positive values for sigma, low_thresh and high_thresh\n");
        exit(1);
    }
    if (argc == 7) {
        *num_runs = strtol(argv[6], nullptr, 10);
    } else {
        *num_runs = 1;
    }
}

#endif //UTILS_H
