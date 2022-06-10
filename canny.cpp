#include "canny.h"

// returns 1 channel flattened image in grayscale
void RGBToGrayscale(const uint8_t* src, uint8_t* dst, int width, int height, int channels) {
    assert(("3 channels are required for conversion to grayscale", channels == 3));

    #pragma omp parallel for default(none) firstprivate(width, height, channels) shared(src, dst)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            const uint8_t* pixel = &src[(i * width + j) * channels];
            dst[i * width + j] = static_cast<uint8_t>(std::min(0.299f * pixel[0] + 0.587f * pixel[1] + 0.114f * pixel[2], 255.f));
        }
    }
}

void ApplyConv2D(const float* src, float* dst, int width, int height, const float* kernel, int kernel_width, int kernel_height) {
    int kernel_radius_width = kernel_width / 2;
    int kernel_radius_height = kernel_height / 2;

    #pragma omp parallel for default(none) firstprivate(width, height, kernel_radius_width, kernel_radius_height, kernel_width) shared(src, dst, kernel)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {

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
}

// dst is assumed to have shape (height, width, 2)
void ApplySeparableConv2D(const float* src, float* dst, float* buffer, int width, int height, const float* kernel_h, int kernel_width, const float* kernel_v, int kernel_height) {
    int kernel_radius_width = kernel_width / 2;
    int kernel_radius_height = kernel_height / 2;

    #pragma omp parallel for default(none) firstprivate(height, width, kernel_radius_height) shared(src, kernel_v, buffer)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {

            float conv_val_v = 0.;
            size_t idx, k_idx;
            for (int k_i = -kernel_radius_height; k_i <= kernel_radius_height; ++k_i) {
                if (i + k_i >= 0 && i + k_i < height) {
                    idx = (i + k_i) * width + j;
                    k_idx = k_i + kernel_radius_height;

                    conv_val_v += src[idx] * kernel_v[k_idx];
                }
            }
            idx = i * width + j;
            buffer[idx] = conv_val_v;
        }
    }

    #pragma omp parallel for default(none) firstprivate(height, width, kernel_radius_width) shared(src, kernel_h, buffer, dst)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {

            float conv_val_h = 0.;

            size_t idx, k_idx;
            for (int k_j = -kernel_radius_width; k_j <= kernel_radius_width; ++k_j) {
                if (j + k_j >= 0 && j + k_j < width) {
                    idx = i * width + j + k_j;
                    k_idx = k_j + kernel_radius_width;

                    conv_val_h += buffer[idx] * kernel_h[k_idx];
                }
            }
            idx = i * width + j;
            dst[idx] = conv_val_h;
        }
    }
}

void MagnitudeAndDirection(const float* horizontal, const float* vertical, float* mag, uint8_t* dir, int width, int height) {
    size_t total_len = width * height;

    #pragma omp parallel for default(none) firstprivate(total_len) shared(horizontal, vertical, mag, dir)
    for (size_t i = 0; i < total_len; ++i) {
        float cur_value = sqrtf(horizontal[i] * horizontal[i] + vertical[i] * vertical[i]);
        mag[i] = cur_value;

        float cur_angle = atanf(vertical[i] / horizontal[i]) + M_PI_2f32;
        if (cur_angle > 3.5f * M_PI_4f32) {
            cur_angle -= 3.5f * M_PI_4f32;
        }
        dir[i] = static_cast<uint8_t>(ceilf(4 * cur_angle * M_1_PIf32 - 0.5f));
    }
}

void NonMaxSuppression(const float* mag, const uint8_t* dir, float* dst, int width, int height) {
    #pragma omp parallel for default(none) firstprivate(height, width) shared(mag, dir, dst)
    for (int i = 1; i < height - 1; ++i) {
        for (int j = 1; j < width - 1; ++j) {

            size_t idx = i * width + j;

            float max_along_grad = 0.0;
            if (dir[idx] == 0) {
                max_along_grad = std::max(mag[(i - 1) * width + (j    )], mag[(i + 1) * width + (j    )]);
            } else if (dir[idx] == 1) {
                max_along_grad = std::max(mag[(i - 1) * width + (j + 1)], mag[(i + 1) * width + (j - 1)]);
            } else if (dir[idx] == 2) {
                max_along_grad = std::max(mag[(i    ) * width + (j - 1)], mag[(i    ) * width + (j + 1)]);
            } else if (dir[idx] == 3) {
                max_along_grad = std::max(mag[(i - 1) * width + (j - 1)], mag[(i + 1) * width + (j + 1)]);
            }

            if (mag[idx] > max_along_grad) {
                dst[idx] = mag[idx];
            } else {
                dst[idx] = 0;
            }
        }
    }
}

void Thresholding(const float* src, uint8_t* dst, int width, int height, float high_thresh, float low_thresh) {
    #pragma omp parallel for default(none) firstprivate(width, height, high_thresh, low_thresh) shared(src, dst)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {

            size_t idx = i * width + j;
            if (src[idx] > high_thresh) {
                dst[idx] = STRONG_EDGE;
            } else if (src[idx] > low_thresh) {
                dst[idx] = WEAK_EDGE;
            } else {
                dst[idx] = NO_EDGE;
            }

        }
    }
}

void Hysteresis(uint8_t* src, int width, int height, int pad) {
    pad = std::max(pad, 1);
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

    bool changes = true;
    while (changes) {
        changes = false;
        #pragma omp parallel for default(none) firstprivate(width, height, pad, offsets) shared(src) reduction(||:changes)
        for (int i = pad; i < height - pad; ++i) {
            for (int j = pad; j < width - pad; ++j) {
                size_t idx = i * width + j;
                if (src[idx] == STRONG_EDGE) {
                    for (int k = 0; k < 8; ++k) {
                        if (src[idx + offsets[k]] == WEAK_EDGE) {
                            src[idx + offsets[k]] = STRONG_EDGE;
                            changes = true;
                        }
                    }
                } else if (src[idx] == WEAK_EDGE) {
                    for (int k = 0; k < 8; ++k) {
                        if (src[idx + offsets[k]] == STRONG_EDGE) {
                            src[idx] = STRONG_EDGE;
                            changes = true;
                        }
                    }
                }
            }
        }
    }

    #pragma omp parallel for default(none) firstprivate(width, height, pad) shared(src)
    for (int i = pad; i < height - pad; ++i) {
        for (int j = pad; j < width - pad; ++j) {
            size_t idx = i * width + j;
            if (src[idx] == WEAK_EDGE) {
                src[idx] = NO_EDGE;
            }
        }
    }
}