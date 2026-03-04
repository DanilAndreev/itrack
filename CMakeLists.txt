cmake_minimum_required(VERSION 4.1)
project(itrack CUDA)

set(CMAKE_CUDA_STANDARD 20)

set(SOURCE_FILES
        src/main.cu
        src/stb_impl.cu
    )
set(HEADER_FILES
        src/pch.cuh
)

add_executable(itrack ${HEADER_FILES} ${SOURCE_FILES})
target_precompile_headers(itrack PRIVATE src/pch.cuh)
set_source_files_properties(${HEADER_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)

set_target_properties(itrack PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(itrack PRIVATE external/stb)

