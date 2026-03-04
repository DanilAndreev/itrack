#include "pch.cuh"
#include <stb_image.h>

namespace Kernels {
    __global__ void kernel() {

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

    float* dImage = nullptr;
    cudaMalloc(&dImage, width * height * channels * sizeof(float));

    cudaMemcpy();


    return 0;
}
