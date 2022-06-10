#include <cstdlib>
#include <iostream>
#include <vector>
#include <array>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"

#include "canny.h"
#include "utils.h"

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

void ApplyCannyCPU(const uint8_t* input, uint8_t* output, int width, int height, int channels, float sigma, float low_thresh, float high_thresh, bool sep_conv=true) {
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

    RGBToGrayscale(input, output, width, height, channels);

    float* padded_float_image = AllocateArray<float>(padded_width, padded_height, 1);
    float* padded_float_h = AllocateArray<float>(padded_width, padded_height, 1);
    float* padded_float_v = AllocateArray<float>(padded_width, padded_height, 1);
    float* padded_float_buffer = AllocateArray<float>(padded_width, padded_height, 1);
    uint8_t* padded_uint = AllocateArray<uint8_t>(padded_width, padded_height, 1);

//    ++++++++++++++
//    uint8_t* test_out = AllocateArray<uint8_t>(padded_width, padded_height, 1);
//    float* test_cut = AllocateArray<float>(padded_width, padded_height, 1);
//    ++++++++++++++

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
//    /////
//    FloatToUint(padded_float_h, test_out, padded_width, padded_height, 1,40);
//    SaveImage(test_out, "../test_h.png", padded_width, padded_height, 1);
//    FloatToUint(padded_float_v, test_out, padded_width, padded_height, 1,40);
//    SaveImage(test_out, "../test_v.png", padded_width, padded_height, 1);
//    /////

    MagnitudeAndDirection(padded_float_h, padded_float_v, padded_float_buffer, padded_uint, padded_width, padded_height);
//    /////
//    FloatToUint(padded_float_buffer, test_out, padded_width, padded_height, 1);
//    SaveImage(test_out, "../test_mag.png", padded_width, padded_height, 1);
//    UintToFloat(padded_uint, test_cut, padded_width, padded_height, 1);
//    FloatToUint(test_cut, test_out, padded_width, padded_height, 1, 63);
//    SaveImage(test_out, "../test_dir.png", padded_width, padded_height, 1);
//    /////

    NonMaxSuppression(padded_float_buffer, padded_uint, padded_float_image, padded_width, padded_height);
//    /////
//    FloatToUint(padded_float_image, test_out, padded_width, padded_height, 1);
//    SaveImage(test_out, "../test_nms.png", padded_width, padded_height, 1);
//    /////
    Thresholding(padded_float_image, padded_uint, padded_width, padded_height, high_thresh, low_thresh);
    Hysteresis(padded_uint, padded_width, padded_height, kernel_radius);

    CentralCropImage(padded_uint, output, padded_width, padded_height, 1, kernel_radius);

    free(kernel_diff);
    free(kernel_norm);

    free(padded_float_image);
    free(padded_float_h);
    free(padded_float_v);
    free(padded_float_buffer);
    free(padded_uint);
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

    long total_duration = 0;
    for (int n_run = 0; n_run < num_runs; ++n_run) {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        ApplyCannyCPU(image, edges, width, height, channels, sigma, low_thresh, high_thresh, true);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Run " << n_run << std::endl;
        std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms] to complete" << std::endl;
        total_duration += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    }
    std::cout << "Mean runtime over " << num_runs << " runs: " << total_duration / num_runs << "[ms]" << std::endl;


    SaveImage(edges, output_path, width, height, 1);

    free(image);
    free(edges);

    return 0;
}