#include "pch.cuh"
#include <stb_image.h>
#include "LayerKernels.cuh"

float* DeviceAllocAndFillFilter(uint filterSize, uint chCount, uint filterCount) noexcept {
    std::vector<float> staging{};
    staging.resize(filterCount * chCount * filterSize * filterSize);

    float *dFilter;
    cudaMalloc(&dFilter, staging.size() * sizeof(float));

    for (size_t fIdx = 0; fIdx < filterSize; ++fIdx) {
        for (size_t chIdx = 0; chIdx < filterSize; ++chIdx) {
            for (size_t y = 0; y < filterSize; ++y) {
                for (size_t x = 0; x < filterSize; ++x) {
                    auto idx = fIdx * (chCount*filterSize*filterSize) + chIdx * (filterSize*filterSize) + y * filterSize + x;
                    staging[idx] = 1.0f;
                }
            }
        }
    }
    cudaMemcpy(dFilter, staging.data(), staging.size() * sizeof(float), cudaMemcpyHostToDevice);
    return dFilter;
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

    for (size_t y = 0; y < height; y += 4) {
        for (size_t x = 0; x < width; x += 2) {
            unsigned char r = img[(y * width + x) * channels + 0];
            unsigned char g = img[(y * width + x) * channels + 1];
            unsigned char b = img[(y * width + x) * channels + 2];

            float v = static_cast<float>(r) / 255.f;
            if (v == 0) {
                printf(" ");
            } else if (v < 0.25f) {
                printf("░");
            } else if (v > 0.25f && v < 0.5f) {
                printf("▒");
            } else if (v > 0.5f && v < 0.75f) {
                printf("▓");
            } else {
                printf("█");
            }
        }
        printf("\n");
    }

    constexpr uint L0_FILTER_COUNT = 96;

    // width, height, chCount, batchCount
    uint4 tDim{ static_cast<uint>(width), static_cast<uint>(height), static_cast<uint>(channels), 1};

    std::vector<float> tensorStaging{};
    tensorStaging.resize(tDim.x * tDim.y * tDim.z * tDim.w);
    assert(tDim.w == 1 && "Batches are not fully supported yet");
    for (size_t y = 0; y < tDim.y; ++y) {
        for (size_t x = 0; x < tDim.x; ++x) {
            for (size_t chIdx = 0; chIdx < tDim.z; ++chIdx) {
                unsigned char v = img[(y * tDim.x + x) * channels + chIdx];
                tensorStaging[chIdx * tDim.x * tDim.y + y * tDim.x + x] = static_cast<float>(v) / 255.0f;
            }
        }
    }

    float* dTensorMemA;
    float* dTensorMemB;

    float* dFilterWL0;

    assert(tDim.w == 1);

    const auto TMP_MEM_EL = tDim.x * tDim.y * 384;
    SUCC(cudaMalloc(&dTensorMemA, TMP_MEM_EL * sizeof(float)));
    SUCC(cudaMalloc(&dTensorMemB, TMP_MEM_EL * sizeof(float)));

    cudaMemcpy(dTensorMemA, tensorStaging.data(), tensorStaging.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dTensorMemB, 0, tDim.x * tDim.y * 384 * sizeof(float));

    cudaMalloc(&dFilterWL0, L0_FILTER_COUNT * tDim.z * 11 * 11 * sizeof(float));

    dFilterWL0 = DeviceAllocAndFillFilter(11, tDim.z, L0_FILTER_COUNT);
    tDim = Layers::Conv2D(tDim, 11, 2, L0_FILTER_COUNT, dFilterWL0, dTensorMemA, dTensorMemB);
    cudaMemset(dTensorMemA, 0, TMP_MEM_EL * sizeof(float));
    Layers::BatchNorm2D(tDim, dTensorMemB);
    tDim = Layers::MaxPool2D(tDim, 3, 2, dTensorMemB, dTensorMemA);
    cudaMemset(dTensorMemB, 0, TMP_MEM_EL * sizeof(float));

    // Layers::ReLU();

    return 0;
}
