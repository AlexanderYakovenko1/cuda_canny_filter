cmake_minimum_required(VERSION 3.22)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.6/bin/nvcc)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

project(my_cuda_canny LANGUAGES CUDA CXX)

find_package(OpenMP)

add_executable(cuda main.cu include/stb_image.h include/stb_image_write.h utils.h canny.cu)
add_executable(openmp main.cpp include/stb_image.h include/stb_image_write.h utils.h canny.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(openmp PUBLIC OpenMP::OpenMP_CXX)
endif()

set_target_properties(cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)