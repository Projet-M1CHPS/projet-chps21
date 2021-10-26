#include "Transform.hpp"

#include <cassert>
#include <iostream>

constexpr unsigned possible_brightness = 255 * 3;
constexpr unsigned half_brigthness = (unsigned)possible_brightness / 3;

namespace image::transform {

namespace {
color_t _greyByBlackAndWhite(const Color &color) {
    return ((unsigned)(((unsigned)color.r) + ((unsigned)color.g) +
                       ((unsigned)color.b))) < half_brigthness
               ? 0
               : 255;
}
color_t _greyByMean(const Color &color) {
    return ((unsigned)(((unsigned)color.r) + ((unsigned)color.g) +
                       ((unsigned)color.b)) /
            3);
}
color_t _greyByDesaturation(const Color &color) {
    color_t max =
        std::max(color.r, std::max(color.g, color.b));  // Optimisation possible
    color_t min = std::min(color.r, std::min(color.g, color.b));  // by grouping
    return (color_t)(max - min) / 2;
}
color_t _greyByMinDecomposition(const Color &color) {
    return std::min(color.r, std::min(color.g, color.b));
}
color_t _greyByMaxDecomposition(const Color &color) {
    return std::max(color.r, std::max(color.g, color.b));
}

void _greyscaling(Image &image, color_t (*func)(const Color &)) {
    for (size_t i = 0; i < image.colors.size(); i++) {
        color_t grey = func(image.colors[i]);
        image.colors[i].r = grey;
        image.colors[i].g = grey;
        image.colors[i].b = grey;
    }
}
std::array<long double, possible_brightness> _createBrightnessHeightmap(
    Image const &image) {
    std::cout << "> _getBrightnessHeightmap()" << std::endl;
    long double increment_value = 1.0 / image.width * image.height;
    std::array<long double, possible_brightness> heightmap;

    heightmap.fill(0.0);
    for (Color each : image.colors) {
        unsigned index = each.r + each.g + each.b;
        heightmap[index] += increment_value;
    }
    std::cout << "< _getBrightnessHeightmap()" << std::endl;
    return heightmap;
}
}  // namespace

bool BlackWhiteScale::transform(Image &image) {
    unsigned mid_value = 255 * 3 / 2;
    _greyscaling(image, _greyByBlackAndWhite);
    return true;
}

bool GreyScale::transform(Image &image) {
    std::cout << "> GreyScale::transform()" << std::endl;
    _greyscaling(image, _greyByMean);
    std::cout << "< GreyScale::transform()" << std::endl;
    return true;
}
}  // namespace image::transform
