#include "Transform.hpp"

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>

constexpr unsigned half_brigthness = (unsigned)possible_brightness / 3;

namespace image::transform {

namespace {
color_t _greyByBinarisation(const RGBColor &color) {
    return ((unsigned)color.r + color.g + color.b) < half_brigthness ? 0U
                                                                     : 255U;
}
color_t _greyByMean(const RGBColor &color) {
    return (((unsigned)color.r) + color.g + color.b) / 3;
}

color_t _greyByDesaturation(const RGBColor &color) {
    color_t max =
        std::max(color.r, std::max(color.g, color.b));  // Optimisation possible
    color_t min = std::min(color.r, std::min(color.g, color.b));  // by grouping
    return (color_t)(max - min) / 2;
}

color_t _greyByMinDecomposition(const RGBColor &color) {
    return std::min(color.r, std::min(color.g, color.b));
}

color_t _greyByMaxDecomposition(const RGBColor &color) {
    return std::max(color.r, std::max(color.g, color.b));
}

void _greyscaling(Image &image, std::function<color_t(const RGBColor &)> fn) {
    /*std::transform(image.cbegin(), image.cend(), image.begin(), [&](Color c) {
        color_t grey = fn(c);
        c = {grey, grey, grey};
    }); // Mathys fait nimp
    */
    for (RGBColor &each : image) {
        color_t grey = fn(each);
        each = {grey, grey, grey};
    }
}

/**
 * @return [rHistogram, gHistogram, bHistogram]
 */
std::vector<std::vector<double>> createColorsHistograms(Image const &image) {
    std::cout << "> createColorsHistograms()" << std::endl;
    double increment_value = 1.0 / ((double)image.getDimension());
    std::vector<std::vector<double>> histograms(3);
    histograms[0].resize(image.getDimension());
    histograms[1].resize(image.getDimension());
    histograms[2].resize(image.getDimension());
    for (RGBColor const &each : image) {
        histograms[0][each.r] += increment_value;
        histograms[1][each.g] += increment_value;
        histograms[2][each.b] += increment_value;
    }
    std::cout << "< createColorsHistograms()" << std::endl;
    return histograms;
}

/**
 * @return [percentage of pixels with brightness = 0; ... = 1; ... =
 * possible_brightness-1]
 */
std::vector<double> createBrightnessHistogram(Image const &image) {
    double increment_value = 1.0 / ((double)image.getDimension());
    std::vector<double> histogram(possible_brightness + 1);

    for (RGBColor each : image)
        histogram[each.getBrightness()] += increment_value;
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
    std::vector<double> brightness_histogram = createBrightnessHistogram(image);

    unsigned median_brightness = (unsigned)possible_brightness / 2;
    double cumul = 0.0;
    for (unsigned i = 0; i < possible_brightness; i++) {
        cumul += brightness_histogram[i];
        if (cumul > .5) {
            median_brightness = (i > 0) ? i - 1 : 0;
            break;
        }
    }
    for (RGBColor &each : image) {
        color_t grey = (each.getBrightness() < median_brightness) ? 0 : 255;
        each = {grey, grey, grey};
    }
    std::cout << "< HistogramBinaryScale::transform()" << std::endl;
    return true;
}

bool HistogramInversion::transform(Image &image) {
    std::cout << "> HistogramInversion::transform()" << std::endl;
    for (RGBColor &each : image)
        each = {(color_t)(255 - each.r), (color_t)(255 - each.g),
                (color_t)(255 - each.b)};
    std::cout << "< HistogramInversion::transform()" << std::endl;
    return true;
}

bool HistogramSpread::transform(Image &image) {
    std::cout << "> HistogramSpread::transform()" << std::endl;
    std::vector<std::vector<double>> histograms = createColorsHistograms(image);
    std::array<size_t, 3> minIndexes;
    minIndexes.fill(0);
    std::array<size_t, 3> maxIndexes;
    maxIndexes.fill(255);
    for (size_t c = 0; c < 3; c++) {  // Channels iteration loop
        for (size_t i = 254; i >= 0; i--)
            if (histograms[c][i] > 0.0) {
                maxIndexes[c] = i + 1;
                break;
            }
        for (size_t i = 1; i < 256; i++)
            if (histograms[c][i] > 0.0) {
                minIndexes[c] = i + 1;
                break;
            }
    }
    std::array<size_t, 3> dMins = {minIndexes[0] - 0, minIndexes[1] - 0,
                                   minIndexes[2] - 0};
    std::array<size_t, 3> dMaxs = {255 - maxIndexes[0], 255 - maxIndexes[1],
                                   255 - maxIndexes[2]};
    std::array<double, 3> coefs = {(dMaxs[0] - dMins[0]) / (255.0),
                                   (dMaxs[1] - dMins[1]) / (255.0),
                                   (dMaxs[2] - dMins[2]) / (255.0)};
    size_t index = 0;
    for (RGBColor &each : image) {
        each.r = coefs[0] * index;
        each.g = coefs[1] * index;
        each.b = coefs[1] * index;
    }
    std::cout << "< HistogramSpread::transform()" << std::endl;
    return true;
}
}  // namespace image::transform
