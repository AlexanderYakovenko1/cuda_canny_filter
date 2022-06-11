#include <cstdlib>
#include <iostream>
#include <utility>
#include <chrono>

#include "canny.h"
#include "utils.h"

std::pair<int64_t, int64_t> ApplyCannyCPU(const uint8_t* input, uint8_t* output, int width, int height, int channels, float sigma, float low_thresh, float high_thresh, bool sep_conv=true) {
    std::chrono::steady_clock::time_point all_begin = std::chrono::steady_clock::now();

    int kernel_radius;
    float* kernel_diff;
    float* kernel_norm;
    if (sep_conv) {
        kernel_diff = GaussianDerivativeKernel1D(sigma, &kernel_radius, true);
        kernel_norm = GaussianDerivativeKernel1D(sigma, &kernel_radius, false);
    } else {
        kernel_diff = GaussianDerivativeKernel2D(sigma, &kernel_radius, true);
        kernel_norm = GaussianDerivativeKernel2D(sigma, &kernel_radius, false);
    }

    int padded_width = width + 2 * kernel_radius;
    int padded_height = height + 2 * kernel_radius;

    float* padded_float_image = AllocateArray<float>(padded_width, padded_height, 1);
    float* padded_float_h = AllocateArray<float>(padded_width, padded_height, 1);
    float* padded_float_v = AllocateArray<float>(padded_width, padded_height, 1);
    float* padded_float_buffer = AllocateArray<float>(padded_width, padded_height, 1);
    uint8_t* padded_uint = AllocateArray<uint8_t>(padded_width, padded_height, 1);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    RGBToGrayscale(input, output, width, height, channels);
    Pad2D<uint8_t, float>(output, padded_float_image, width, height, 1, kernel_radius);

    if (sep_conv) {
        ApplySeparableConv2D(padded_float_image, padded_float_h, padded_float_buffer, padded_width, padded_height,
                             kernel_diff, 2 * kernel_radius + 1, kernel_norm, 2 * kernel_radius + 1);
        ApplySeparableConv2D(padded_float_image, padded_float_v, padded_float_buffer, padded_width, padded_height,
                             kernel_norm, 2 * kernel_radius + 1, kernel_diff, 2 * kernel_radius + 1);
    } else {
        ApplyConv2D(padded_float_image, padded_float_h, padded_width, padded_height, kernel_diff, 2 * kernel_radius + 1, 2 * kernel_radius + 1);
        ApplyConv2D(padded_float_image, padded_float_v, padded_width, padded_height, kernel_norm, 2 * kernel_radius + 1, 2 * kernel_radius + 1);
    }

    MagnitudeAndDirection(padded_float_h, padded_float_v, padded_float_buffer, padded_uint, padded_width, padded_height);
    NonMaxSuppression(padded_float_buffer, padded_uint, padded_float_image, padded_width, padded_height);
    Thresholding(padded_float_image, padded_uint, padded_width, padded_height, high_thresh, low_thresh);
    Hysteresis(padded_uint, padded_width, padded_height, kernel_radius);

    CentralCropImage(padded_uint, output, padded_width, padded_height, 1, kernel_radius);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    free(kernel_diff);
    free(kernel_norm);

    free(padded_float_image);
    free(padded_float_h);
    free(padded_float_v);
    free(padded_float_buffer);
    free(padded_uint);

    std::chrono::steady_clock::time_point all_end = std::chrono::steady_clock::now();

    return std::make_pair(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count(),
                          std::chrono::duration_cast<std::chrono::microseconds>(all_end - all_begin).count());
}

int main(int argc, char** argv) {
    char* input_path;
    char* output_path;
    float sigma, low_thresh, high_thresh;
    int num_runs;


    ParseArguments(argc, argv, &input_path, &output_path, &sigma, &low_thresh, &high_thresh, &num_runs);
    printf("Applying Canny filter to %s with sigma=%f, low_thresh=%f, high_thresh=%f and saving to %s\n", input_path, sigma, low_thresh, high_thresh, output_path);

    int width, height, channels;
    uint8_t* image = stbi_load(input_path, &width, &height, &channels, STBI_rgb);
    uint8_t* edges = AllocateArray<uint8_t>(width, height, 1);

    int64_t total_duration_with = 0;
    int64_t total_duration_without = 0;
    for (int n_run = 0; n_run < num_runs; ++n_run) {
        auto [without_allocs, with_allocs] = ApplyCannyCPU(image, edges, width, height, channels, sigma, low_thresh, high_thresh, true);
        std::cout << "Run " << n_run << std::endl;
        std::cout << "Took " << without_allocs << "[µs] to complete (without allocation)" << std::endl;
        std::cout << "Took " << with_allocs << "[µs] to complete (with allocation)" << std::endl;
        total_duration_without += without_allocs;
        total_duration_with += with_allocs;
    }
    std::cout << "Mean runtime over " << num_runs << " runs: " << total_duration_without / num_runs << "[µs] (without allocation)" << std::endl;
    std::cout << "Mean runtime over " << num_runs << " runs: " << total_duration_with / num_runs << "[µs] (with allocation)" << std::endl;

    std::cout << total_duration_without / num_runs << " " << total_duration_with / num_runs << std::endl;

    SaveImage(edges, output_path, width, height, 1);

    free(image);
    free(edges);

    return 0;
}