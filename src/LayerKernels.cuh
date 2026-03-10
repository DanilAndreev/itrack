#pragma once

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

    __global__ void Conv2D(uint4 dim, uint stride, const float* filter, const float* srcTensor, float* dstTensor);

    __global__ void MaxPool2DWrp(uint4 dim, uint windowSize, uint stride, float* srcTensor, float* dstTensor);

    __global__ void BatchMean2D(uint4 dim, const float* tensor, float* outMean);
    __global__ void BatchVariance2D(uint4 dim, const float* tensor, const float* mean, float* outVariance);
    __global__ void BatchNorm2D(uint4 dim, float* tensor, const float* mean, const float* variance);
}

namespace Layers {
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

    void ReLU(uint4 dim, float* tensor);
    uint4 Conv2D(uint4 dim, uint filterSize, uint stride, uint filterCount, const float* filter, const float* srcTensor, float* dstTensor);
    uint4 MaxPool2D(uint4 dim, uint windowSize, uint stride, float* srcTensor, float* dstTensor);
    void BatchNorm2D(uint4 dim, float* tensor);
}