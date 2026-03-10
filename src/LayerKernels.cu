#include "pch.cuh"
#include "LayerKernels.cuh"

namespace Kernels {
    __global__ void ReLU(uint4 dim, float* tensor) {
        uint batchIdx = blockIdx.z / dim.z;
        uint chIdx = blockIdx.z % dim.z;
        uint elY = blockIdx.y * blockDim.y + threadIdx.y;
        uint elX = blockIdx.x * blockDim.x + threadIdx.x;

        if (elX >= dim.x || elY >= dim.y)
            return;

        const uint linearIdx = batchIdx * (dim.z * dim.y * dim.x) + chIdx * (dim.y * dim.x) + elY * dim.x + elX;
        float value = tensor[linearIdx];
        tensor[linearIdx] = value > 0.0f ? value : 0.0f;
    }

    //TODO: Naive implementation. Currently it wastes a lot of lanes per warp. Will be optimized later.
    __global__ void Conv2D(uint4 dim, uint stride, const float* filter, const float* srcTensor, float* dstTensor) {
        uint filterIdx = blockIdx.z / dim.z;
        uint chIdx = blockIdx.z % dim.z;
        uint elY = blockIdx.y * stride + threadIdx.y;
        uint elX = blockIdx.x * stride + threadIdx.x;


        const uint srcLinearIdx = chIdx * (dim.y * dim.x) + elY * dim.x + elX;

        const uint filterSize = blockDim.x;
        assert(blockDim.x == blockDim.y);

        const uint filterLinearIdx = filterIdx * (dim.z*filterSize*filterSize) + chIdx * (filterSize*filterSize) + threadIdx.y * filterSize + threadIdx.x;
        const float filterVal = filter[filterLinearIdx];
        float weighted = srcTensor[srcLinearIdx] * filterVal;
        atomicAdd(&dstTensor[filterIdx * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x], weighted);
    }

    // Naive implementation that underutilizes warp threads and has unoptimal memory pattern. Will be optimized in the future.
    __global__ void MaxPool2DWrp(uint4 dim, uint windowSize, uint stride, float* srcTensor, float* dstTensor) {
        assert(blockDim.x <= warpSize);
        // uint batchIdx = blockIdx.z / chCount;
        uint chIdx = blockIdx.z % dim.z;

        uint2 windowIdx = {threadIdx.x % windowSize, threadIdx.x / windowSize};
        uint elY = blockIdx.y * stride + windowIdx.x;
        uint elX = blockIdx.x * stride + windowIdx.y;

        uint linearIdx = chIdx * (dim.y * dim.x) + elY * dim.x + elX;

        float maxVal = threadIdx.x < windowSize*windowSize
                           ? srcTensor[linearIdx]
                           : FLT_MIN;
        for (int i = blockDim.x / 2; i >= 1; i /= 2) {
            maxVal = max(__shfl_xor_sync(0xffffffff, maxVal, i, 32), maxVal);
        }

        if (threadIdx.x == 0)
            dstTensor[chIdx * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x] = maxVal;
    }

    __global__ void BatchMean2D(uint4 dim, const float* tensor, float* outMean) {
        uint batchTotalElements = dim.w * dim.x * dim.y;

        uint batchIdx = blockIdx.z / dim.z;
        uint chIdx = blockIdx.z % dim.z;
        uint elY = blockIdx.y * blockDim.y + threadIdx.y;
        uint elX = blockIdx.x * blockDim.x + threadIdx.x;

        const uint linearIdx = batchIdx * (dim.z * dim.y * dim.x) + chIdx * (dim.y * dim.x) + elY * dim.x + elX;
        float value = tensor[linearIdx];
        atomicAdd(&outMean[chIdx], value / static_cast<float>(batchTotalElements));
    }

    __global__ void BatchVariance2D(uint4 dim, const float* tensor, const float* mean, float* outVariance) {
        uint batchTotalElements = dim.w * dim.x * dim.y;

        uint batchIdx = blockIdx.z / dim.z;
        uint chIdx = blockIdx.z % dim.z;
        uint elY = blockIdx.y * blockDim.y + threadIdx.y;
        uint elX = blockIdx.x * blockDim.x + threadIdx.x;

        const uint linearIdx = batchIdx * (dim.z * dim.y * dim.x) + chIdx * (dim.y * dim.x) + elY * dim.x + elX;
        float value = tensor[linearIdx];
        float vm = value - mean[chIdx];
        atomicAdd(&outVariance[chIdx], (vm * vm) / static_cast<float>(batchTotalElements));
    }

    __global__ void BatchNorm2D(uint4 dim, float* tensor, const float* mean, const float* variance) {
        uint batchIdx = blockIdx.z / dim.z;
        uint chIdx = blockIdx.z % dim.z;
        uint elY = blockIdx.y * blockDim.y + threadIdx.y;
        uint elX = blockIdx.x * blockDim.x + threadIdx.x;

        const uint linearIdx = batchIdx * (dim.z * dim.y * dim.x) + chIdx * (dim.y * dim.x) + elY * dim.x + elX;
        float value = tensor[linearIdx];
        constexpr float EPS = 1e-5;
        float var = variance[chIdx];
        tensor[linearIdx] = (value - mean[chIdx]) / sqrt(var + EPS);
    }
}

namespace Layers {
    void ReLU(uint4 dim, float* tensor) {
        uint3 groupsize = {8, 8, 1};
        uint3 gridsize = {IntDivideCeil(dim.x, groupsize.x), IntDivideCeil(dim.y, groupsize.y), dim.z * dim.w};
        Kernels::ReLU<<<gridsize, groupsize>>>(dim, tensor);
    }

    uint4 Conv2D(uint4 dim, uint filterSize, uint stride, uint filterCount, const float* filter, const float* srcTensor, float* dstTensor) {
        uint3 groupsize = {filterSize, filterSize, 1};
        uint3 gridsize = {dim.x / stride, dim.y / stride , dim.z * filterCount};

        Kernels::Conv2D<<<gridsize, groupsize>>>(dim, stride, filter, srcTensor, dstTensor);
        return {gridsize.x, gridsize.y, filterCount, dim.w};
    }

    uint4 MaxPool2D(uint4 dim, uint windowSize, uint stride, float* srcTensor, float* dstTensor) {
        assert(windowSize <= 8 && "Only single warp dimension are supported. (max 32 thr)");
        assert(dim.x >= windowSize);
        assert((dim.x - windowSize) % stride == 0);
        assert(dim.y >= windowSize);
        assert((dim.y - windowSize) % stride == 0);

        uint groupsize = CeilToMultipleOf(windowSize*windowSize, 16u);
        uint3 gridsize = {(dim.x - windowSize) / stride + 1, (dim.y - windowSize) / stride + 1, dim.z};
        Kernels::MaxPool2DWrp<<<gridsize, groupsize>>>(dim, windowSize, stride, srcTensor, dstTensor);
        return {gridsize.x, gridsize.y, dim.z, dim.w};
    }

    void BatchNorm2D(uint4 dim, float* tensor) {
        //TODO: move outside and reuse
        float* dScratch;
        const uint scratchSizeInBytes = dim.z * 2 * sizeof(dScratch[0]);
        cudaMalloc(&dScratch, scratchSizeInBytes);
        cudaMemset(dScratch, 0, scratchSizeInBytes);

        float* dMean = &dScratch[0];
        float* dVariance = &dScratch[dim.z];

        std::vector<float> readback{};
        readback.resize(scratchSizeInBytes / sizeof(float));


        uint3 groupsize = {8, 8, 1};
        uint3 gridsize = {IntDivideCeil(dim.x, groupsize.x), IntDivideCeil(dim.y, groupsize.y), dim.w * dim.z};
        Kernels::BatchMean2D<<<gridsize, groupsize>>>(dim, tensor, dMean);

        cudaMemcpy(readback.data(), dScratch, scratchSizeInBytes, cudaMemcpyDeviceToHost);
        printf("\nScratch Mean: ");
        for (auto v : readback) {
            printf("%f ", v);
        }

        Kernels::BatchVariance2D<<<gridsize, groupsize>>>(dim, tensor, dMean, dVariance);

        cudaMemcpy(readback.data(), dScratch, scratchSizeInBytes, cudaMemcpyDeviceToHost);
        printf("\nScratch Variance: ");
        for (auto v : readback) {
            printf("%f ", v);
        }

        Kernels::BatchNorm2D<<<gridsize, groupsize>>>(dim, tensor, dMean, dVariance);
    }
}