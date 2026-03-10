#pragma once

namespace Utils {
    template<class T>
    void Print4D(T* tensor, uint batchCount, uint chCount, uint2 dim) {
        for (size_t bIdx = 0; bIdx < batchCount; ++bIdx) {

            for (size_t chIdx = 0; chIdx < chCount; ++chIdx) {
                std::cout << "\n\nBatch[" << bIdx << "] Ch[" << chIdx << "]:\n";
                for (size_t y = 0; y < dim.y; ++y) {
                    for (size_t x = 0; x < dim.x; ++x) {
                        auto idx = bIdx * (chCount * dim.y * dim.x) + chIdx * (dim.y * dim.x) + y * dim.x + x;
                        std::cout << tensor[idx];
                    }
                    std::cout << "\n";
                }
            }
        }
        std::cout << std::endl;
    }
}