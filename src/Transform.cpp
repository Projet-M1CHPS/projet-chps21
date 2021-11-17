#include "Transform.hpp"

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <math.h>

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
              std::max(r, std::max(g, b));   // Optimisation possible
      grayscale_t min =
              std::min(r, std::min(g, b));   // by grouping
      return (grayscale_t) (max - min) / 2;
    }

    grayscale_t _grayByMinDecomposition(const unsigned &r, const unsigned &g, const unsigned &b) {
      return std::min(r, std::min(g, b));
    }

    grayscale_t _grayByMaxDecomposition(const unsigned &r, const unsigned &g, const unsigned &b) {
      return std::max(r, std::max(g, b));
    }
  }   // namespace color_to_gray

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

  Resize::Resize(size_t width, size_t height) : width(width), height(height){};

  /**
   * @brief Sub-function for resizing (stable on both up-scaling and down-scaling, but not really efficient on up-scaling)
   *
   * @param factors Pair: (source.width-1)/(dest.width-1) ; (source.height-1)/(dest.height-1)
   * @param source source image
   * @param dest destination image
   */
  void downscaling(std::pair<double, double> factors, GrayscaleImage const &source, GrayscaleImage &dest) {
    for (size_t x = 0; x < dest.getWidth(); x++) {
      for (size_t y = 0; y < dest.getHeight(); y++) {
        if (x == 0 || y == 0 || x == dest.getWidth() - 1 || y == dest.getHeight() - 1) {
          size_t index = (x == 0 ? 0 : source.getWidth() - 1) + source.getWidth() * (y == 0 ? 0 : source.getHeight() - 1);
          dest.getData()[x + y * dest.getWidth()] = source.getData()[index];   // borders
        } else {
          size_t orig_left_x = (size_t) std::floor(factors.first * x);
          size_t orig_right_x = (size_t) std::ceil(factors.first * x);
          size_t orig_up_y = (size_t) std::floor(factors.second * y);
          size_t orig_down_y = (size_t) std::ceil(factors.second * y);
          grayscale_t value = (grayscale_t)
                  std::round((source.getData()[orig_left_x + source.getWidth() * orig_up_y] +
                              source.getData()[orig_left_x + source.getWidth() * orig_down_y] +
                              source.getData()[orig_right_x + source.getWidth() * orig_up_y] +
                              source.getData()[orig_right_x + source.getWidth() * orig_down_y]) /
                             4.0);
          dest.getData()[x + dest.getWidth() * y] = value;
        }
      }
    }
  }

  bool Resize::transform(GrayscaleImage &image) {
    const image::GrayscaleImage source = image;   // In order to modify the image ref, we first need to copy it.
    image.setSize(width, height);
    std::pair<double, double> factors((double) (source.getWidth() - 1) / (width - 1), (double) (source.getHeight() - 1) / (height - 1));
    // Works for both up & down scaling.
    // A upscaling(...) function will be defined soon to enhance that particular case.
    downscaling(factors, source, image);
    return true;
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
}   // namespace image::transform
