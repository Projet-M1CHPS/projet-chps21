#include "Transform.hpp"

#include <cassert>
#include <functional>
#include <iostream>

constexpr unsigned half_brigthness = (unsigned)possible_brightness / 3;

namespace image::transform {

namespace {
color_t _greyByBinarisation(const Color &color) {
    return ((unsigned)color.r + color.g + color.b) < half_brigthness ? 0U
                                                                     : 255U;
}
color_t _greyByMean(const Color &color) {
    return (((unsigned)color.r) + color.g + color.b) / 3;
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

void _greyscaling(Image &image, std::function<color_t(const Color &)> fn) {
    /*std::transform(image.cbegin(), image.cend(), image.begin(), [&](Color c) {
        color_t grey = fn(c);
        c = {grey, grey, grey};
    }); // Mathys fait nimp
    */
    for (Color &each : image) {
        color_t grey = fn(each);
        each = {grey, grey, grey};
    }
}

/**
 * @return [rHistogram, gHistogram, bHistogram]
 */
std::vector<std::vector<double>> _createColorsHistograms(Image const &image) {
    std::cout << "> _createColorsHistograms()" << std::endl;
    double increment_value = 1.0 / ((double)image.getDimension());
    std::vector<std::vector<double>> histograms(3);
    histograms[0].resize(image.getDimension());
    histograms[1].resize(image.getDimension());
    histograms[2].resize(image.getDimension());
    for (Color const &each : image) {
        histograms[0][each.r] += increment_value;
        histograms[1][each.g] += increment_value;
        histograms[2][each.b] += increment_value;
    }
    std::cout << "< _createColorsHistograms()" << std::endl;
    return histograms;
}

std::vector<double> _createBrightnessHistogram(Image const &image) {
    double increment_value = 1.0 / ((double)image.getDimension());
    std::vector<double> histogram(possible_brightness + 1);

    for (Color each : image)
        histogram[each.r + each.g + each.b] += increment_value;
    return histogram;
}
}  // namespace

bool NoTransform::transform(Image &image) { return true; }

bool BinaryScale::transform(Image &image) {
    _greyscaling(image, _greyByBinarisation);
    return true;
}

bool GreyScale::transform(Image &image) {
    std::cout << "> GreyScale::transform()" << std::endl;
    _greyscaling(image, _greyByMean);
    std::cout << "< GreyScale::transform()" << std::endl;
    return true;
}

bool HistogramBinaryScale::transform(Image &image) {
    std::cout << "> HistogramBinaryScale::transform()" << std::endl;
    std::vector<double> brightness_histogram =
        _createBrightnessHistogram(image);

    unsigned median_brightness = (unsigned)possible_brightness / 2;
    long double cumul = 0.0;
    for (unsigned i = 0; i < possible_brightness; i++) {
        cumul += brightness_histogram[i];
        if (cumul > .5) {
            median_brightness = (i > 0) ? i - 1 : 0;
            break;
        }
    }
    for (Color &each : image) {
        color_t grey =
            ((each.r + each.g + each.b) < median_brightness) ? 0 : 255;
        each = {grey, grey, grey};
    }
    std::cout << "< HistogramBinaryScale::transform()" << std::endl;
    return true;
}

bool HistogramInversion::transform(Image &image) {
    std::cout << "> HistogramInversion::transform()" << std::endl;
    for (Color &each : image)
        each = {(color_t)(255 - each.r), (color_t)(255 - each.g),
                (color_t)(255 - each.b)};
    std::cout << "< HistogramInversion::transform()" << std::endl;
    return true;
}
}  // namespace image::transform
