#include "Transform.hpp"

#include <cassert>
#include <iostream>

using namespace image;

std::array<long double, POSSIBLE_BRIGHTNESSES> _createBrightnessHeightmap(
    Image const &image) {
    std::cout << "> _getBrightnessHeightmap()" << std::endl;
    long double increment_value = 1.0 / image.width * image.height;
    std::array<long double, POSSIBLE_BRIGHTNESSES> heightmap;

    heightmap.fill(0.0);
    for (Color each : image.colors) {
        unsigned index = each.r + each.g + each.b;
        heightmap[index] += increment_value;
    }
    std::cout << "< _getBrightnessHeightmap()" << std::endl;
    return heightmap;
}

bool transform::BlackWhiteScale::transform(Image &image) {
    std::array<long double, POSSIBLE_BRIGHTNESSES> heightmap =
        _createBrightnessHeightmap(image);
    unsigned long dimension = image.width * image.height;

    for (unsigned i = 0; i < dimension; i++) {
        color_t height_colorized = 255 * ((color_t)heightmap[i]);
        Color current(height_colorized, height_colorized, height_colorized);
        image.colors[i] = current;
    }
    return true;
}

bool transform::GreyScale::transform(Image &image) {
    std::cout << "> transform::GreyScale::transform()" << std::endl;
    unsigned long dimension = image.width * image.height;

    for (unsigned i = 0; i < dimension; i++) {
        unsigned mean = (unsigned) ((image.colors[i].r+image.colors[i].g+image.colors[i].b)/3.0);
        if(mean < 0)
            mean = 0;
        else if(mean > 255)
            mean = 255;
        image.colors[i].r = mean;
        image.colors[i].g = mean;
        image.colors[i].b = mean;
    }
    //image.print();
    std::cout << "< transform::GreyScale::transform()" << std::endl;
    return true;
}