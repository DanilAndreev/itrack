#pragma once

namespace Utils {
    template<class T>
    void Print4D(T* tensor, uint4 dim) {
        for (size_t bIdx = 0; bIdx < dim.w; ++bIdx) {
            for (size_t chIdx = 0; chIdx < dim.z; ++chIdx) {
                std::cout << "\n\nBatch[" << bIdx << "] Ch[" << chIdx << "]:\n";
                for (size_t y = 0; y < dim.y; ++y) {
                    for (size_t x = 0; x < dim.x; ++x) {
                        auto idx = bIdx * (dim.z * dim.y * dim.x) + chIdx * (dim.y * dim.x) + y * dim.x + x;
                        std::cout << tensor[idx] << " ";
                    }
                    std::cout << "\n";
                }
                if (chIdx == 2) break;
            }
        }
        std::cout << std::endl;
    }

    template<class T>
    void ReadbackAndPrint4D(T* dTensor, uint4 dim) {
        std::vector<T> readback{};
        const size_t linearElCount = dim.w * dim.z * dim.y * dim.x;
        readback.resize(linearElCount);
        cudaMemcpy(readback.data(), dTensor, linearElCount * sizeof(T), cudaMemcpyDeviceToHost);
        Print4D(readback.data(), dim);
    }
}