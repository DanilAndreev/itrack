#include "pch.cuh"
#include <stb_image.h>


using uint = uint32_t;

namespace Kernels {
    template<bool FirstPass, class T, T MaxT = std::numeric_limits<T>::max(), T MinT = std::numeric_limits<T>::min()>
    __global__ void MinMaxSumReduce(T* input, T* outputSum, T* scratchReduce, uint inputElCount) {
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
        if (FirstPass) {
            atomicAdd(&outputSum[0], minVal);
        }
        for (uint i = 16; i >= 1; i /= 2) {
            minVal = min(__shfl_xor_sync(0xffffffff, minVal, i, 32), minVal);
            maxVal = max(__shfl_xor_sync(0xffffffff, maxVal, i, 32), maxVal);
        }
        scratchReduce[blockIdx.x * 2 + 0] = minVal;
        scratchReduce[blockIdx.x * 2 + 1] = maxVal;
    }

    __global__ void Conv2D(uint batchCount, uint chCount, uint2 dim, uint stride, const float* filter, const float* srcTensor, float* dstTensor) {
        // uint2 dtID = {blockIdx.x * (blockDim.x + stride) + threadIdx.x, blockIdx.y * (blockDim.y + stride) + threadIdx.y};

        //TODO: add filterCount dimension

        uint batchIdx = blockIdx.z / chCount;
        uint chIdx = blockIdx.z % chCount;
        uint elY = blockIdx.y * stride + threadIdx.y;
        uint elX = blockIdx.x * stride + threadIdx.x;


        const uint linearIdx = batchIdx * (chCount * dim.y * dim.x) + chIdx * (dim.y * dim.x) + elY * dim.x + elX;

        const uint filterSizeX = blockDim.x;
        float weighted = srcTensor[linearIdx] * filter[threadIdx.y * filterSizeX + threadIdx.x];
        atomicAdd(&dstTensor[batchIdx * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x], weighted);
    }

    // Naive implementation that underutilizes warp threads and has unoptimal memory pattern. Will be optimized in the future.
    __global__ void MaxPool2DWrp(const float* src, uint srcDimX, float* dst, uint windowDim, uint stride) {
        assert(blockDim.x <= warpSize);

        uint2 windowIdx = {threadIdx.x % windowDim, threadIdx.x / windowDim};
        uint2 srcIdx = {blockIdx.x * stride + windowIdx.x, blockIdx.y * stride + windowIdx.y};

        float maxVal = threadIdx.x < windowDim*windowDim
                           ? src[srcIdx.y * srcDimX + srcIdx.x]
                           : FLT_MIN;
        for (int i = blockDim.x / 2; i >= 1; i /= 2) {
            maxVal = max(__shfl_xor_sync(0xffffffff, maxVal, i, 32), maxVal);
        }

        if (threadIdx.x == 0)
            dst[blockIdx.y * gridDim.x + blockIdx.x] = maxVal;
    }

    __global__ void BatchMean2D(uint batchCount, uint chCount, uint2 dim, const float* tensor, float* outMean) {
        uint batchTotalElements = batchCount * dim.x * dim.y;

        uint batchIdx = blockIdx.z / chCount;
        uint chIdx = blockIdx.z % chCount;
        uint elY = blockIdx.y * blockDim.y + threadIdx.y;
        uint elX = blockIdx.x * blockDim.x + threadIdx.x;

        const uint linearIdx = batchIdx * (chCount * dim.y * dim.x) + chIdx * (dim.y * dim.x) + elY * dim.x + elX;
        float value = tensor[linearIdx];
        atomicAdd(&outMean[chIdx], value / static_cast<float>(batchTotalElements));
    }

    __global__ void BatchVariance2D(uint batchCount, uint chCount, uint2 dim, const float* tensor, const float* mean, float* outVariance) {
        uint batchTotalElements = batchCount * dim.x * dim.y;

        uint batchIdx = blockIdx.z / chCount;
        uint chIdx = blockIdx.z % chCount;
        uint elY = blockIdx.y * blockDim.y + threadIdx.y;
        uint elX = blockIdx.x * blockDim.x + threadIdx.x;

        const uint linearIdx = batchIdx * (chCount * dim.y * dim.x) + chIdx * (dim.y * dim.x) + elY * dim.x + elX;
        float value = tensor[linearIdx];
        float vm = value - mean[chIdx];
        atomicAdd(&outVariance[chIdx], (vm * vm) / static_cast<float>(batchTotalElements));
    }

    __global__ void BatchNorm2D(uint batchCount, uint chCount, uint2 dim, float* tensor, const float* mean, const float* variance) {
        uint batchIdx = blockIdx.z / chCount;
        uint chIdx = blockIdx.z % chCount;
        uint elY = blockIdx.y * blockDim.y + threadIdx.y;
        uint elX = blockIdx.x * blockDim.x + threadIdx.x;

        const uint linearIdx = batchIdx * (chCount * dim.y * dim.x) + chIdx * (dim.y * dim.x) + elY * dim.x + elX;
        float value = tensor[linearIdx];
        constexpr float EPS = 1e-5;
        float var = variance[chIdx];
        tensor[linearIdx] = (value - mean[chIdx]) / sqrt(var + EPS);
    }
}


template <class T>
void Reduce(T* input, T* outputSum, T* scratchReduce, size_t inputElCount) {
    constexpr size_t GROUPSIZE = 5;
    size_t prevGc = inputElCount;
    size_t gc = IntDivideCeil(prevGc, GROUPSIZE);
    T* scratchA = scratchReduce;
    const auto offB = CeilToMultipleOf(gc * 2, 32ull);
    T* scratchB = scratchA + offB;
    T* stepInput = input;
    do {
        if (stepInput == input) {
            Kernels::MinMaxSumReduce<true><<<gc, GROUPSIZE>>>(stepInput, outputSum, scratchA, prevGc);
        } else {
            Kernels::MinMaxSumReduce<false><<<gc, GROUPSIZE>>>(stepInput, outputSum, scratchA, prevGc);
        }
        stepInput = scratchA;
        T* temp = scratchA;
        scratchA = scratchB;
        scratchB = temp;
        prevGc = gc;
        gc = IntDivideCeil(gc, GROUPSIZE);
    } while (gc > 1);
    Kernels::MinMaxSumReduce<false><<<1, GROUPSIZE>>>(stepInput, outputSum, scratchA, prevGc);
}

void BatchNorm2D(uint batchCount, uint chCount, uint2 dim, float* tensor) {
    //TODO: move outside and reuse
    float* dScratch;
    const uint scratchSizeInBytes = chCount * 2 * sizeof(dScratch[0]);
    cudaMalloc(&dScratch, scratchSizeInBytes);
    cudaMemset(dScratch, 0, scratchSizeInBytes);

    float* dMean = &dScratch[0];
    float* dVariance = &dScratch[chCount];

    std::vector<float> readback{};
    readback.resize(scratchSizeInBytes / sizeof(float));


    uint3 groupsize = {8, 8, 1};
    uint3 gridsize = {IntDivideCeil(dim.x, groupsize.x), IntDivideCeil(dim.y, groupsize.y), batchCount * chCount};
    Kernels::BatchMean2D<<<gridsize, groupsize>>>(batchCount, chCount, dim, tensor, dMean);

    cudaMemcpy(readback.data(), dScratch, scratchSizeInBytes, cudaMemcpyDeviceToHost);
    printf("\nScratch Mean: ");
    for (auto v : readback) {
        printf("%f ", v);
    }

    Kernels::BatchVariance2D<<<gridsize, groupsize>>>(batchCount, chCount, dim, tensor, dMean, dVariance);

    cudaMemcpy(readback.data(), dScratch, scratchSizeInBytes, cudaMemcpyDeviceToHost);
    printf("\nScratch Variance: ");
    for (auto v : readback) {
        printf("%f ", v);
    }

    Kernels::BatchNorm2D<<<gridsize, groupsize>>>(batchCount, chCount, dim, tensor, dMean, dVariance);
}

void MaxPool2D(float* src, uint2 srcDim, float* dst, uint windowDim, uint stride) {
    assert(windowDim <= 8 && "Only single warp dimension are supported. (max 32 thr)");
    assert(srcDim.x >= windowDim);
    assert((srcDim.x - windowDim) % stride == 0);
    assert(srcDim.y >= windowDim);
    assert((srcDim.y - windowDim) % stride == 0);

    uint3 gridsize = {(srcDim.x - windowDim) / stride + 1, (srcDim.y - windowDim) / stride + 1, 1};
    uint groupsize = CeilToMultipleOf(windowDim*windowDim, 16u);
    Kernels::MaxPool2DWrp<<<gridsize, groupsize>>>(src, srcDim.x, dst, windowDim, stride);
}

void Print4D(float* tensor, uint batchCount, uint chCount, uint2 dim) {
    for (size_t bIdx = 0; bIdx < batchCount; ++bIdx) {
        printf("\n\n Batch[%lld]:\n", bIdx);
        for (size_t chIdx = 0; chIdx < chCount; ++chIdx) {
            printf("\n\n Batch[%lld] Ch[%lld]:\n", bIdx, chIdx);
            for (size_t y = 0; y < dim.y; ++y) {
                for (size_t x = 0; x < dim.x; ++x) {
                    auto idx = bIdx * (chCount * dim.y * dim.x) + chIdx * (dim.y * dim.x) + y * dim.x + x;
                    printf("%f, ", tensor[idx]);
                }
                printf("\n");
            }
        }
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

    if (false) {
        const uint WINDOW_DIM = 3;
        const uint STRIDE = 2;
        uint2 tensorDim = {17, 17};
        uint32_t tensorElCount = tensorDim.x * tensorDim.y;

        std::vector<float> tensor{};
        tensor.resize(tensorElCount);
        for (size_t i = 0; i < tensorElCount; ++i) {
            tensor[i] = float(i) + 2;

            if (i % tensorDim.x == 0)
                printf("\n");
            printf("%f ", tensor[i]);
        }
        printf("\n\n");

        float* dSrcTensor;
        float* dDstTensor;

        uint2 dstDim = {(tensorDim.x - WINDOW_DIM) / STRIDE + 1, (tensorDim.y - WINDOW_DIM) / STRIDE + 1};
        uint dstElCount = dstDim.x * dstDim.y;
        cudaMalloc(&dSrcTensor, tensorElCount * sizeof(float));
        cudaMalloc(&dDstTensor, dstElCount * sizeof(float));
        cudaMemcpy(dSrcTensor, tensor.data(), tensorElCount * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(dDstTensor, 0, dstElCount * sizeof(float));

        MaxPool2D(dSrcTensor, tensorDim, dDstTensor, WINDOW_DIM, STRIDE);

        cudaDeviceSynchronize();

        std::vector<float> result{};
        result.resize(dstElCount);
        cudaMemcpy(result.data(), dDstTensor, dstElCount * sizeof(float), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < dstElCount; ++i) {
            if (i % dstDim.x == 0)
                printf("\n");
            printf("%f ", result[i]);
        }
    }

    if (false) {
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
    }

    if (true) {
        // z = channelsCount
        constexpr uint3 IMG_DIM = {8, 8, 2};
        constexpr size_t BATCH_COUNT = 1;
        constexpr size_t BATCH_EL_COUNT = {IMG_DIM.x * IMG_DIM.y * IMG_DIM.z};

        std::vector<float> tensor{};
        tensor.resize(BATCH_COUNT * BATCH_EL_COUNT);

        float* dTensor;

        srand(NULL);
        for (size_t bIdx = 0; bIdx < BATCH_COUNT; ++bIdx) {
            for (size_t chIdx = 0; chIdx < IMG_DIM.z; ++chIdx) {
                for (size_t y = 0; y < IMG_DIM.y; ++y) {
                    for (size_t x = 0; x < IMG_DIM.x; ++x) {
                        const auto idx = bIdx * (IMG_DIM.z * IMG_DIM.y * IMG_DIM.x) + chIdx * (IMG_DIM.y * IMG_DIM.x) + y * IMG_DIM.x + x;
                        tensor[idx] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                    }
                }
            }
        }
        Print4D(tensor.data(), BATCH_COUNT, IMG_DIM.z, {IMG_DIM.x, IMG_DIM.y});

        cudaMalloc(&dTensor, BATCH_COUNT * BATCH_EL_COUNT * sizeof(float));
        cudaMemcpy(dTensor, tensor.data(), BATCH_COUNT * BATCH_EL_COUNT * sizeof(float), cudaMemcpyHostToDevice);


        BatchNorm2D(BATCH_COUNT, IMG_DIM.z, {IMG_DIM.x, IMG_DIM.y}, dTensor);
        cudaDeviceSynchronize();

        printf("\n\n---------------------------------------------------\n");


        cudaMemcpy(tensor.data(), dTensor, BATCH_COUNT * BATCH_EL_COUNT * sizeof(float), cudaMemcpyDeviceToHost);
        Print4D(tensor.data(), BATCH_COUNT, IMG_DIM.z, {IMG_DIM.x, IMG_DIM.y});
    }

    return 0;
}
