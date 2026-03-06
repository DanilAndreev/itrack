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

    __global__ void Conv2D(const float* input, uint sizeX, uint sizeY, uint stride, const float* filter, float* output) {
        uint2 dtID = {blockIdx.x * (blockDim.x + stride) + threadIdx.x, blockIdx.y * (blockDim.y + stride) + threadIdx.y};
        if (dtID.x > sizeX && dtID.y < sizeY)
            return;
        const uint filterSizeX = blockDim.x;
        float weighted = input[dtID.y * sizeX + dtID.x] * filter[threadIdx.y * filterSizeX + threadIdx.x];
        atomicAdd(&output[blockIdx.y * gridDim.x + blockIdx.x], weighted);
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

    __global__ void BatchMean2D(uint batchCount, uint chCount, uint2 dim, float* const* batches, float* outMean) {
        uint batchTotalElements = batchCount * dim.x * dim.y;

        uint batchIdx = blockIdx.z / chCount;
        uint chIdx = blockIdx.z % chCount;
        uint elY = blockIdx.y * blockDim.y + threadIdx.y;
        uint elX = blockIdx.x * blockDim.x + threadIdx.x;

        float value = batches[batchIdx][chIdx * (dim.y * dim.x) + elY * dim.x + elX];
        atomicAdd(&outMean[chIdx], value / static_cast<float>(batchTotalElements));
    }

    __global__ void BatchVariance2D(uint batchCount, uint chCount, uint2 dim, float* const* batches, const float* mean, float* outVariance) {
        uint batchTotalElements = batchCount * dim.x * dim.y;

        uint batchIdx = blockIdx.z / chCount;
        uint chIdx = blockIdx.z % chCount;
        uint elY = blockIdx.y * blockDim.y + threadIdx.y;
        uint elX = blockIdx.x * blockDim.x + threadIdx.x;

        float value = batches[batchIdx][chIdx * (dim.y * dim.x) + elY * dim.x + elX];
        float vm = value - mean[chIdx];
        atomicAdd(&outVariance[chIdx], (vm * vm) / static_cast<float>(batchTotalElements));
    }

    __global__ void BatchNorm2D(uint batchCount, uint chCount, uint2 dim, float** batches, const float* variance) {
        uint batchIdx = blockIdx.z / chCount;
        uint chIdx = blockIdx.z % chCount;
        uint elY = blockIdx.y * blockDim.y + threadIdx.y;
        uint elX = blockIdx.x * blockDim.x + threadIdx.x;

        uint elIdx = chIdx * (dim.y * dim.x) + elY * dim.x + elX;
        float value = batches[batchIdx][elIdx];
        float chVariance = variance[chIdx];
        constexpr float EPS = 0.00000000001;
        batches[batchIdx][elIdx] = (value - chVariance) / sqrt(chVariance + EPS);
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

void BatchNorm2D(uint batchCount, uint chCount, uint2 dim, float** batches) {
    //TODO: move outside and reuse
    float* dScratch;
    const uint scratchSizeInBytes = chCount * 2 * sizeof(dScratch[0]);
    cudaMalloc(&dScratch, scratchSizeInBytes);
    cudaMemset(dScratch, 0, scratchSizeInBytes);

    float* dMean = &dScratch[0];
    float* dVariance = &dScratch[chCount];

    uint3 groupsize = {8, 8, 1};
    uint3 gridsize = {IntDivideCeil(dim.x, groupsize.x), IntDivideCeil(dim.y, groupsize.y), batchCount * chCount};
    Kernels::BatchMean2D<<<gridsize, groupsize>>>(batchCount, chCount, dim, batches, dMean);
    Kernels::BatchVariance2D<<<gridsize, groupsize>>>(batchCount, chCount, dim, batches, dMean, dVariance);
    Kernels::BatchNorm2D<<<gridsize, groupsize>>>(batchCount, chCount, dim, batches, dVariance);
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
        constexpr uint3 IMG_DIM = {64, 64, 3};
        constexpr size_t BATCH_OCCUPANCY_IN_ELS = {IMG_DIM.x * IMG_DIM.y * IMG_DIM.z};

        std::vector<float*> batches{};
        batches.emplace_back(static_cast<float*>(malloc(BATCH_OCCUPANCY_IN_ELS * sizeof(float))));
        batches.emplace_back(static_cast<float*>(malloc(BATCH_OCCUPANCY_IN_ELS * sizeof(float))));

        std::vector<float*> stagingDBatches{};
        stagingDBatches.resize(batches.size());
        float** dBatches;
        cudaMalloc(&dBatches, batches.size() * sizeof(dBatches[0]));
        for (size_t bIdx = 0; bIdx < batches.size(); ++bIdx) {
            cudaMalloc(&stagingDBatches[bIdx], BATCH_OCCUPANCY_IN_ELS * sizeof(float));
            for (size_t chIdx = 0; chIdx < IMG_DIM.z; ++chIdx) {
                for (size_t y = 0; y < IMG_DIM.y; ++y) {
                    for (size_t x = 0; x < IMG_DIM.x; ++x) {
                        batches[bIdx][chIdx * (IMG_DIM.y * IMG_DIM.x) + y * IMG_DIM.x + x] = float(y) / float(x + 1);
                    }
                }
            }
            cudaMemcpy(stagingDBatches[bIdx], batches[bIdx], BATCH_OCCUPANCY_IN_ELS * sizeof(float), cudaMemcpyHostToDevice);
        }
        cudaMemcpy(dBatches, stagingDBatches.data(), stagingDBatches.size() * sizeof(dBatches[0]), cudaMemcpyHostToDevice);

        BatchNorm2D(batches.size(), IMG_DIM.z, {IMG_DIM.x, IMG_DIM.y}, dBatches);
        cudaDeviceSynchronize();

        printf("\n\n---------------------------------------------------\n");

        for (size_t bIdx = 0; bIdx < batches.size(); ++bIdx) {
            cudaMemcpy(batches[bIdx], stagingDBatches[bIdx], BATCH_OCCUPANCY_IN_ELS * sizeof(float), cudaMemcpyDeviceToHost);

            printf("\n\n Batch[%lld]:\n", bIdx);
            for (size_t chIdx = 0; chIdx < IMG_DIM.z; ++chIdx) {
                printf("\n\n Batch[%lld] Ch[%lld]:\n", bIdx, chIdx);
                for (size_t y = 0; y < IMG_DIM.y; ++y) {
                    for (size_t x = 0; x < IMG_DIM.x; ++x) {
                        batches[bIdx][chIdx * (IMG_DIM.y * IMG_DIM.x) + y * IMG_DIM.x + x] = float(y) / float(x + 1);
                        printf("%f ", batches[bIdx][chIdx * (IMG_DIM.y * IMG_DIM.x) + y * IMG_DIM.x + x]);
                    }
                    printf("\n");
                }
            }
        }

    }

    return 0;
}
