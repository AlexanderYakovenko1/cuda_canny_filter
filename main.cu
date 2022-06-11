
#include "utils.h"
#include "canny.cuh"

constexpr int BLOCK_SIZE = 32;
constexpr int MAX_THREADS = 1024;

std::pair<int64_t, int64_t> ApplyCannyGPU(const uint8_t* input, uint8_t* output, int width, int height, int channels, float sigma, float low_thresh, float high_thresh) {
    std::chrono::steady_clock::time_point all_begin = std::chrono::steady_clock::now();

    int kernel_radius;
    float* kernel_diff;
    float* kernel_norm;

    kernel_diff = GaussianDerivativeKernel1D(sigma, &kernel_radius, true);
    kernel_norm = GaussianDerivativeKernel1D(sigma, &kernel_radius, false);

    int padded_width = width + 2 * kernel_radius;
    int padded_height = height + 2 * kernel_radius;

    dim3 numBlocks(padded_width / BLOCK_SIZE + 1, padded_height / BLOCK_SIZE + 1);
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);

    uint8_t* cuda_input = nullptr;
    uint8_t* cuda_output = nullptr;
    float* cuda_padded_float_image = nullptr;
    float* cuda_padded_float_h = nullptr;
    float* cuda_padded_float_v = nullptr;
    float* cuda_padded_float_buffer = nullptr;
    uint8_t* cuda_padded_uint = nullptr;

    float* cuda_kernel_diff = nullptr;
    float* cuda_kernel_norm = nullptr;

    CHECK_CUDA_ERRS( cudaMalloc(&cuda_input, width * height * channels * sizeof(uint8_t)) );
    CHECK_CUDA_ERRS( cudaMalloc(&cuda_output, width * height * sizeof(uint8_t)) );
    CHECK_CUDA_ERRS( cudaMalloc(&cuda_padded_float_image, padded_width * padded_height * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMalloc(&cuda_padded_float_h, padded_width * padded_height * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMalloc(&cuda_padded_float_v, padded_width * padded_height * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMalloc(&cuda_padded_float_buffer, padded_width * padded_height * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMalloc(&cuda_padded_uint, padded_width * padded_height * sizeof(uint8_t)) );

    CHECK_CUDA_ERRS( cudaMemcpy(cuda_input, input, width * height * channels * sizeof(uint8_t), cudaMemcpyHostToDevice) );

    CHECK_CUDA_ERRS( cudaMalloc(&cuda_kernel_diff, (2 * kernel_radius + 1) * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMalloc(&cuda_kernel_norm, (2 * kernel_radius + 1) * sizeof(float)) );

    CHECK_CUDA_ERRS( cudaMemcpy(cuda_kernel_diff, kernel_diff, (2 * kernel_radius + 1) * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA_ERRS( cudaMemcpy(cuda_kernel_norm, kernel_norm, (2 * kernel_radius + 1) * sizeof(float), cudaMemcpyHostToDevice) );

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    RGBToGrayscale<<<(width * height) / MAX_THREADS + 1, MAX_THREADS>>>(cuda_input, cuda_output, width, height, channels);

    Pad2DGPU<<<numBlocks, threadsPerBlock>>>(cuda_output, cuda_padded_float_image, width, height, 1, kernel_radius);


    int shared_memory_size = CalculateAlignedOffset((2 * kernel_radius + 1)) + (BLOCK_SIZE + 2 * kernel_radius) * (BLOCK_SIZE + 2 * kernel_radius);

    ApplySeparableConv2DRows__shared<<<numBlocks, threadsPerBlock, shared_memory_size * sizeof(float)>>>(
            cuda_padded_float_image, cuda_padded_float_buffer, padded_width, padded_height, cuda_kernel_diff, 2 * kernel_radius + 1);
    ApplySeparableConv2DCols__shared<<<numBlocks, threadsPerBlock, shared_memory_size * sizeof(float)>>>(
            cuda_padded_float_buffer, cuda_padded_float_h, padded_width, padded_height, cuda_kernel_norm, 2 * kernel_radius + 1);
    ApplySeparableConv2DRows__shared<<<numBlocks, threadsPerBlock, shared_memory_size * sizeof(float)>>>(
            cuda_padded_float_image, cuda_padded_float_buffer, padded_width, padded_height, cuda_kernel_norm, 2 * kernel_radius + 1);
    ApplySeparableConv2DCols__shared<<<numBlocks, threadsPerBlock, shared_memory_size * sizeof(float)>>>(
            cuda_padded_float_buffer, cuda_padded_float_v, padded_width, padded_height, cuda_kernel_diff, 2 * kernel_radius + 1);

    MagnitudeAndDirection<<<(padded_width * padded_height) / MAX_THREADS + 1, MAX_THREADS>>>(
            cuda_padded_float_h, cuda_padded_float_v, cuda_padded_float_buffer, cuda_padded_uint, padded_width, padded_height);

    NonMaxSuppression__nobranch<<<numBlocks, threadsPerBlock>>>(
            cuda_padded_float_buffer, cuda_padded_uint, cuda_padded_float_image, padded_width, padded_height);
    Thresholding__basic<<<(padded_width * padded_height) / MAX_THREADS + 1, MAX_THREADS>>>(
            cuda_padded_float_image, cuda_padded_uint, padded_width, padded_height, high_thresh, low_thresh);

    bool host_hysteresis_changes = true;
    while (host_hysteresis_changes) {
        host_hysteresis_changes = false;
        cudaMemcpyToSymbol(device_hysteresis_changes, &host_hysteresis_changes, sizeof(host_hysteresis_changes));

        Hysteresis__basic<<<numBlocks, threadsPerBlock>>>(
                cuda_padded_uint, padded_width, padded_height, kernel_radius);

        cudaMemcpyFromSymbol(&host_hysteresis_changes, device_hysteresis_changes, sizeof(host_hysteresis_changes));
    }

    Cleanup<<<(padded_width * padded_height) / MAX_THREADS + 1, MAX_THREADS>>>(cuda_padded_uint, padded_width, padded_height);

    CentralCropImageGPU<<<numBlocks, threadsPerBlock>>>(cuda_padded_uint, cuda_output, padded_width, padded_height, 1, kernel_radius);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    CHECK_CUDA_ERRS( cudaMemcpy(output, cuda_output, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost) );

    free(kernel_diff);
    free(kernel_norm);

    CHECK_CUDA_ERRS( cudaFree(cuda_input) );
    CHECK_CUDA_ERRS( cudaFree(cuda_output) );
    CHECK_CUDA_ERRS( cudaFree(cuda_padded_float_image) );
    CHECK_CUDA_ERRS( cudaFree(cuda_padded_float_h) );
    CHECK_CUDA_ERRS( cudaFree(cuda_padded_float_v) );
    CHECK_CUDA_ERRS( cudaFree(cuda_padded_float_buffer) );
    CHECK_CUDA_ERRS( cudaFree(cuda_padded_uint) );

    CHECK_CUDA_ERRS( cudaFree(cuda_kernel_diff) );
    CHECK_CUDA_ERRS( cudaFree(cuda_kernel_norm) );

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
        auto [without_allocs, with_allocs] = ApplyCannyGPU(image, edges, width, height, channels, sigma, low_thresh, high_thresh);
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