#include "pch.cuh"
#include <stb_image.h>
#include "LayerKernels.cuh"

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

    // width, height, chCount, batchCount
    uint4 tensorDim{ static_cast<uint>(width), static_cast<uint>(height), static_cast<uint>(channels), 1};

    std::vector<float> tensorStaging{};
    tensorStaging.resize(tensorDim.x * tensorDim.y * tensorDim.z * tensorDim.w);
    assert(tensorDim.w == 1 && "Batches are not fully supported yet");
    for (size_t y = 0; y < tensorDim.y; ++y) {
        for (size_t x = 0; x < tensorDim.x; ++x) {
            for (size_t chIdx = 0; chIdx < tensorDim.z; ++chIdx) {
                unsigned char v = img[(y * tensorDim.x + x) * channels + chIdx];
                tensorStaging[chIdx * tensorDim.x * tensorDim.y + y * tensorDim.x + x] = static_cast<float>(v) / 255.0f;
            }
        }
    }

    float* dInputTensor;

    cudaMalloc(&dInputTensor, tensorDim.x * tensorDim.y * tensorDim.z * tensorDim.w);
    cudaMemcpy(dInputTensor, tensorStaging.data(), tensorStaging.size() * sizeof(float), cudaMemcpyHostToDevice);


    Layers::Conv2D(tensorDim, 11, 2, c, dFilterL0, dInputTensor, dDst);
    Layers::BatchNorm2D(outDim, dDst);
    Layers::MaxPool2D(outDim, 3, 2, dDst, dDst2);
    // Layers::ReLU();

    return 0;
}
