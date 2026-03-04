#include "pch.h"
#include <stb_image.h>

int main() {
    std::cout << "Hello, World!" << std::endl;

    int width, height, channels;
    unsigned char *img = stbi_load("dataset/pixel_art.png", &width, &height, &channels, 0);
    if (img == NULL) {
        printf("Error in loading the image\n");
        exit(1);
    }
    printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, channels);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            unsigned char r = img[(y * width + x) * channels + 0];
            unsigned char g = img[(y * width + x) * channels + 1];
            unsigned char b = img[(y * width + x) * channels + 2];

            float v = static_cast<float>(r) / 255.f;
            if (v == 0) {
                printf(".");
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

    return 0;
}
