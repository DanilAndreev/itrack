#include "pch.cuh"
#include <stb_image.h>


using uint = uint32_t;

namespace Kernels {
    template<bool FirstPass, class T, T MaxT = std::numeric_limits<T>::max(), T MinT = std::numeric_limits<T>::min()>
    __global__ void MaxReduce(T* input, T* outputSum, T* scratchReduce, uint inputElCount) {
        T minVal = MaxT;
        T maxVal = MinT;
        if (FirstPass) {
            uint inIdx = blockIdx.x * blockDim.x + threadIdx.x;
            if (inIdx < inputElCount) {
                minVal = input[inIdx];
                maxVal = minVal;
            }
        } else {
            uint inIdx = blockIdx.x * blockDim.x * 2 + threadIdx.x * 2;
            if (inIdx + 1 < inputElCount * 2) {
                minVal = input[inIdx + 0];
                maxVal = input[inIdx + 1];
            }
        }
        atomicAdd(&outputSum[0], minVal);
        for (uint i = 16; i >= 1; i /= 2) {
            minVal = min(__shfl_xor_sync(0xffffffff, minVal, i, 32), minVal);
            maxVal = max(__shfl_xor_sync(0xffffffff, maxVal, i, 32), maxVal);
        }
        scratchReduce[blockIdx.x * 2 + 0] = minVal;
        scratchReduce[blockIdx.x * 2 + 1] = maxVal;
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


template <class T>
void Reduce(T* input, T* outputSum, T* scratchReduce, size_t inputElCount) {
    constexpr size_t GROUPSIZE = 32;
    size_t prevGc = inputElCount;
    size_t gc = IntDivideCeil(prevGc, GROUPSIZE);
    T* scratchA = scratchReduce;
    const auto offB = CeilToMultipleOf(gc * 2, 32ull);
    T* scratchB = scratchA + offB;
    T* stepInput = input;
    do {
        if (stepInput == input) {
            Kernels::MaxReduce<true><<<gc, GROUPSIZE>>>(stepInput, outputSum, scratchA, prevGc);
        } else {
            Kernels::MaxReduce<false><<<gc, GROUPSIZE>>>(stepInput, outputSum, scratchA, prevGc);
        }
        stepInput = scratchA;
        T* temp = scratchA;
        scratchA = scratchB;
        scratchB = temp;
        prevGc = gc;
        gc = IntDivideCeil(gc, GROUPSIZE);
    } while (gc > 1);
    Kernels::MaxReduce<false><<<1, GROUPSIZE>>>(stepInput, outputSum, scratchA, prevGc);
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

    std::vector<float> array{};
    array.resize(1024);
    for (size_t i = 0; i < array.size(); ++i) {
        array[i] = static_cast<float>(i + 2);
    }

    float* dArray = nullptr;
    float* dSums = nullptr;
    float* dScratch = nullptr;

    cudaMalloc(&dArray, array.size() * sizeof(array[0]));
    cudaMemcpy(dArray, array.data(), array.size() * sizeof(array[0]), cudaMemcpyHostToDevice);

    cudaMalloc(&dSums, array.size() * sizeof(array[0]));
    cudaMemset(dSums, 0, array.size() * sizeof(array[0]));

    cudaMalloc(&dScratch, array.size() * sizeof(array[0]));
    cudaMemset(dScratch, 0, array.size() * sizeof(array[0]));


    float* dImage = nullptr;
    cudaMalloc(&dImage, width * height * channels * sizeof(float));

    Reduce(dArray, dSums, dScratch, array.size());

    cudaDeviceSynchronize();

    std::vector<float> result{};
    result.resize(array.size());
    cudaMemcpy(result.data(), dScratch, result.size() * sizeof(result[0]), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < array.size(); ++i) {
        if (i % 32 == 0)
            printf("\n%lld | ", i / 32);
        printf("%f ", result[i]);
    }

    return 0;
}
