#include "Transform.hpp"

#include <cassert>
#include <iostream>

constexpr unsigned half_brigthness = (unsigned)possible_brightness / 3;

namespace image::transform {

namespace grey_subfunctions {
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
}  // namespace grey_subfunctions

}  // namespace grey_subfunctions
namespace histogram_subfunctions {
/**
 * @return [rHistogram, gHistogram, bHistogram]
 */
std::array<std::array<long double, nb_colors>, 3> _createColorsHistograms(
    Image const &image) {
    std::cout << "> _createColorsHistograms()" << std::endl;
    long double increment_value = 1.0 / ((double)nb_colors);
    std::array<std::array<long double, nb_colors>, 3> histograms;
    histograms[0].fill(0.0);
    histograms[1].fill(0.0);
    histograms[2].fill(0.0);
    for (Color each : image.colors) {
        histograms[0][each.r] += increment_value;
        histograms[1][each.g] += increment_value;
        histograms[2][each.b] += increment_value;
    }
    std::cout << "< _createColorsHistograms()" << std::endl;
    return histograms;
}

std::array<long double, possible_brightness> _createBrightnessHistogram(
    Image const &image) {
    std::cout << "> _createBrightnessHistogram()" << std::endl;
    long double increment_value = 1.0 / ((double)possible_brightness);
    std::array<std::array<long double, nb_colors>, 3> histograms =
        _createColorsHistograms(image);
    std::array<long double, possible_brightness> histogram;
    histogram.fill(0.0);
    for (unsigned i = 0; i < nb_colors; i++)
        histogram[histograms[0][i] + histograms[1][i] + histograms[2][i]] +=
            increment_value;

    std::cout << "< _createBrightnessHistogram()" << std::endl;
    return histogram;
}
}  // namespace histogram_subfunctions

bool BlackWhiteScale::transform(Image &image) {
    unsigned mid_value = 255 * 3 / 2;
    grey_subfunctions::_greyscaling(image,
                                    grey_subfunctions::_greyByBlackAndWhite);
    return true;
}

bool GreyScale::transform(Image &image) {
    std::cout << "> GreyScale::transform()" << std::endl;
    grey_subfunctions::_greyscaling(image, grey_subfunctions::_greyByMean);
    std::cout << "< GreyScale::transform()" << std::endl;
    return true;
}

bool HistogramInversion::transform(Image &image) {
    std::cout << "> HistogramInversion::transform()" << std::endl;
    for (Color &each : image.colors) {
        each.r = 255 - each.r;
        each.g = 255 - each.g;
        each.b = 255 - each.b;
    }
    std::cout << "< HistogramInversion::transform()" << std::endl;
    return true;
}
}  // namespace image::transform
