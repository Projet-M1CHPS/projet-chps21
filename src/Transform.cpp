#include "Transform.hpp"

#include <cassert>
#include <iostream>

using namespace image;

void _getBrightnessHeightmap(Image const &image, long double (&heightmap)[POSSIBLE_BRIGHTNESSES]) {
    long double increment_value = 1.0 / image.width * image.height;

    // Testing the initialization of colors_count = [0;0;0;...]
    for (unsigned i = 0; i < POSSIBLE_BRIGHTNESSES; i++) {
        assert(heightmap[i] == 0.0);
    }

    for (Color each : image.colors) {
        heightmap[each.r + each.g + each.b] += increment_value;
    }
}

Image* transform::BlackWhiteScale::transform(Image const &image) {
    long double heightmap[POSSIBLE_BRIGHTNESSES] = {0.0};
    _getBrightnessHeightmap(image, (&heightmap)[POSSIBLE_BRIGHTNESSES]);
    unsigned long dimension = image.width * image.height;
    
    std::vector<Color> new_colors;
    for (unsigned i = 0; i < dimension; i++) {
        color_t height_colorized = 255 * ((color_t)heightmap[i]);
        Color current(height_colorized, height_colorized, height_colorized);
        new_colors.push_back(current);
    }
    Image modified(image.width, image.height, new_colors);
    Image* ret_modified = &modified;
    return ret_modified;
}

Image *transform::GreyScale::transform(Image const &image) {
    long double heightmap[POSSIBLE_BRIGHTNESSES] = {0.0};
    _getBrightnessHeightmap(image, (&heightmap)[POSSIBLE_BRIGHTNESSES]);
    unsigned long dimension = image.width * image.height;

    std::vector<Color> new_colors;
    for (unsigned i = 0; i < dimension; i++) {
        color_t height_colorized = (color_t)(heightmap[i] * 255);
        Color current(height_colorized, height_colorized, height_colorized);
        new_colors.push_back(current);
    }
    Image modified(image.width, image.height, new_colors);
    Image* ret_modified = &modified;
    return ret_modified;
}