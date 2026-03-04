cmake_minimum_required(VERSION 4.1)
project(itrack CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(itrack main.cu)

set_target_properties(itrack PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(itrack PRIVATE external/stb)

