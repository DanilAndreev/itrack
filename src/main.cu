#include "pch.cuh"
#include <stb_image.h>

using uint = uint32_t;

namespace Kernels {
    template<class T>
    __global__ void MaxReduce(T* input) {
        T value = input[threadIdx.x];
        for (uint i = 16; i >= 1; i /= 2) {
            value = max(__shfl_xor_sync(0xffffffff, value, i, 32), value);
        }
        printf("Thread %d final value = %d\n", threadIdx.x, value);
    }

    __global__ void Conv2D(const float* input, uint sizeX, uint sizeY, uint stride, const float* filter, float* output) {
        uint2 dtID = {blockIdx.x * (blockDim.x + stride) + threadIdx.x, blockIdx.y * (blockDim.y + stride) + threadIdx.y};
        if (dtID.x > sizeX && dtID.y < sizeY)
            return;
        const uint filterSizeX = blockDim.x;
        float weighted = input[dtID.y * sizeX + dtID.x] * filter[threadIdx.y * filterSizeX + threadIdx.x];
        atomicAdd(&output[blockIdx.y * gridDim.x + blockIdx.x], weighted);
    }

    __global__ void MaxPool2D(const float* input, uint sizeX, uint sizeY, uint stride, float* output) {
        uint2 dtID = {blockIdx.x * (blockDim.x + stride) + threadIdx.x, blockIdx.y * (blockDim.y + stride) + threadIdx.y};
        if (dtID.x > sizeX && dtID.y < sizeY)
            return;
        const uint filterSizeX = blockDim.x;
        float weighted = input[dtID.y * sizeX + dtID.x];
        // atomicMax(&output[blockIdx.y * gridDim.x + blockIdx.x], weighted);
    }
}

int main() {
    std::cout << "Hello, World!" << std::endl;

    int width, height, channels;
    unsigned char *img = stbi_load("dataset/cat.png", &width, &height, &channels, 0);
    if (img == NULL) {
        printf("Error in loading the image\n");
        exit(1);
    }
    printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, channels);

    // for (size_t y = 0; y < height; y += 4) {
    //     for (size_t x = 0; x < width; x += 2) {
    //         unsigned char r = img[(y * width + x) * channels + 0];
    //         unsigned char g = img[(y * width + x) * channels + 1];
    //         unsigned char b = img[(y * width + x) * channels + 2];
    //
    //         float v = static_cast<float>(r) / 255.f;
    //         if (v == 0) {
    //             printf(" ");
    //         } else if (v < 0.25f) {
    //             printf("░");
    //         } else if (v > 0.25f && v < 0.5f) {
    //             printf("▒");
    //         } else if (v > 0.5f && v < 0.75f) {
    //             printf("▓");
    //         } else {
    //             printf("█");
    //         }
    //     }
    //     printf("\n");
    // }

    float* dImage = nullptr;
    cudaMalloc(&dImage, width * height * channels * sizeof(float));

    Kernels::MinReduce<<<1, 32>>>();

    return 0;
}
