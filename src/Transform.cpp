#include "Transform.hpp"

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>

namespace image::transform {

  namespace color_to_gray {

    grayscale_t _grayByBinarisation(const unsigned &r, const unsigned &g, const unsigned &b) {
      return (r + g + b) < (max_brightness * 3) / 2 ? 0U : 255U;
    }
    grayscale_t _grayByMean(const unsigned &r, const unsigned &g, const unsigned &b) {
      return (r + g + b) / 3;
    }

    grayscale_t _grayByDesaturation(const unsigned &r, const unsigned &g, const unsigned &b) {
      grayscale_t max =
              std::max(r, std::max(g, b));// Optimisation possible
      grayscale_t min =
              std::min(r, std::min(g, b));// by grouping
      return (grayscale_t) (max - min) / 2;
    }

    grayscale_t _grayByMinDecomposition(const unsigned &r, const unsigned &g, const unsigned &b) {
      return std::min(r, std::min(g, b));
    }

    grayscale_t _grayByMaxDecomposition(const unsigned &r, const unsigned &g, const unsigned &b) {
      return std::max(r, std::max(g, b));
    }
  }// namespace color_to_gray

  /**
  * @return [percentage of pixels with brightness = 0; ... = 1; ... =
  * max_brightness-1]
  */
  std::vector<double> createHistogram(GrayscaleImage const &image) {
    double increment_value = 1.0 / ((double) image.getSize());
    std::vector<double> histogram(nb_colors);
    std::for_each(image.begin(), image.end(), [&histogram, increment_value](auto e) { histogram[e] += increment_value; });
    return histogram;
  }

  bool BinaryScale::transform(GrayscaleImage &image) {
    float half_brightness = max_brightness / 2.0;
    std::for_each(image.begin(), image.end(), [half_brightness](auto &e) { e = e < half_brightness ? 0U : 255U; });
    return true;
  }

  bool BinaryScaleByMedian::transform(GrayscaleImage &image) {
    std::vector<double> brightness_histogram = createHistogram(image);

    grayscale_t median_brightness = 0;
    double cumul = 0.0;
    for (size_t i = 0; i < max_brightness; i++) {
      cumul += brightness_histogram[i];
      if (cumul > .5) {
        median_brightness = (grayscale_t) ((i > 0) ? i - 1 : 0);
        break;
      }
    }
    std::for_each(image.begin(), image.end(), [median_brightness](auto &e) { e = e < median_brightness ? 0U : 255U; });
    return true;
  }

  bool Inversion::transform(GrayscaleImage &image) {
    std::for_each(image.begin(), image.end(), [](auto &e) { e = max_brightness - e; });
    return true;
  }
}// namespace image::transform
