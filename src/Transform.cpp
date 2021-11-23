#include "Transform.hpp"

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <math.h>
#include <numeric>

namespace image::transform {

  Crop::Crop(size_t width, size_t height, size_t orig_x, size_t orig_y) : width(width), height(height), orig_x(orig_x), orig_y(orig_y){};

  Resize::Resize(size_t width, size_t height) : width(width), height(height){};

  float _get2DVectorNorm(size_t x, size_t y) {
    return sqrtf(x * x + y * y);
  }

  /**
   * @brief Sub-function for resizing (stable on both up-scaling and down-scaling, but not really efficient on up-scaling)
   *
   * @param factors Pair: (source.width-1)/(dest.width-1) ; (source.height-1)/(dest.height-1)
   * @param source source image
   * @param dest destination image
   */
  void downscaling(std::pair<double, double> factors, GrayscaleImage const &source, GrayscaleImage &dest) {
    for (size_t x = 0; x < dest.getWidth(); x++) {
      size_t source_x = std::round(x * factors.first);
      dest(x, 0) = source(source_x, 0);                                           // top row
      dest(x, dest.getHeight() - 1) = source(source_x, source.getHeight() - 1);   // bottom row
    }

    for (size_t y = 0; y < dest.getHeight(); y++) {
      size_t source_y = std::round(y * factors.second);
      dest(0, y) = source(0, source_y);                                         // left col
      dest(dest.getWidth() - 1, y) = source(source.getWidth() - 1, source_y);   // right col
    }

    for (size_t x = 1; x < dest.getWidth() - 1; x++) {
      for (size_t y = 1; y < dest.getHeight() - 1; y++) {
        size_t orig_left_x = (size_t) std::floor(factors.first * x);
        size_t orig_right_x = (size_t) std::ceil(factors.first * x);
        size_t orig_up_y = (size_t) std::floor(factors.second * y);
        size_t orig_down_y = (size_t) std::ceil(factors.second * y);
        float top_left_coef = _get2DVectorNorm(orig_left_x - x, orig_up_y - y);
        float top_right_coef = _get2DVectorNorm(orig_right_x - x, orig_up_y - y);
        float bottom_left_coef = _get2DVectorNorm(orig_left_x - x, orig_down_y - y);
        float bottom_right_coef = _get2DVectorNorm(orig_right_x - x, orig_down_y - y);

        grayscale_t mean = (grayscale_t)
                std::round((source(orig_left_x, orig_up_y) +
                            source(orig_left_x, orig_down_y) +
                            source(orig_right_x, orig_up_y) +
                            source(orig_right_x, orig_down_y)) /
                           4.0);
        dest(x, y) = mean;
      }
    }
  }

  bool Crop::transform(GrayscaleImage &image) {
    if (orig_x < 0 || orig_y < 0 || width <= 0 || height <= 0) {
      std::cout << "Crop can not be applied on this image: new origin needs to be greater or equal than (0,0) and new dimensions needs to be strictly greater than (0,0)" << std::endl;
      return false;
    } else if (orig_x + width > image.getWidth()) {
      std::cout << "Crop can not be applied on this image: orig_x[" << orig_x << "] + width[" << width << "] > image.width[" << image.getWidth() << "]" << std::endl;
      return false;
    } else if (orig_y + height > image.getHeight()) {
      std::cout << "Crop can not be applied on this image: orig_y[" << orig_y << "] + height[" << height << "] > image.height[" << image.getHeight() << "]" << std::endl;
      return false;
    }
    const image::GrayscaleImage source = image;   // In order to modify the image ref, we first need to copy it.
    image.setSize(width, height);
    for (size_t x = orig_x; x < orig_x + width; x++) {
      for (size_t y = orig_y; y < orig_y + height; y++) {
        image(x - orig_x, y - orig_y) = source(x, y);
      }
    }
    return true;
  }

  bool Resize::transform(GrayscaleImage &image) {
    if (width <= 0 || height <= 0) {
      std::cout << "Resize can not be applied on this image: new dimensions needs to be strictly greater than (0,0)" << std::endl;
      return false;
    } else if (height == image.getHeight() && width == image.getWidth())   // Case where we don't need to resize the image.
      return false;
    const image::GrayscaleImage source = image;   // In order to modify the image ref, we first need to copy it.
    image.setSize(width, height);
    std::pair<double, double> factors((double) (source.getWidth() - 1) / (width - 1), (double) (source.getHeight() - 1) / (height - 1));
    // Works for both up & down scaling.
    // A upscaling(...) function will be defined soon to enhance that particular case.
    downscaling(factors, source, image);
    return true;
  }

  bool Equalize::transform(GrayscaleImage &image) {
    std::vector<double> ratio_histogram = image.createRatioHistogram();

    std::vector<grayscale_t> associations(nb_colors);
    double cumulated_ratios = 0;
    for (size_t i = 0; i < nb_colors; i++) {
      cumulated_ratios += ratio_histogram[i];
      associations[i] = (grayscale_t) std::round(cumulated_ratios * max_brightness);
    }

    std::for_each(image.begin(), image.end(), [associations](auto &e) { e = associations[e]; });

    return true;
  }

  void binaryScalingByCap(GrayscaleImage &image, grayscale_t cap) {
    std::for_each(image.begin(), image.end(), [cap](auto &e) { e = e < cap ? 0U : 255U; });
  }

  bool BinaryScale::transform(GrayscaleImage &image) {
    float half_brightness = max_brightness / 2.0;
    binaryScalingByCap(image, half_brightness);
    return true;
  }

  bool BinaryScaleByMedian::transform(GrayscaleImage &image) {
    std::vector<double> brightness_histogram = image.createRatioHistogram();

    grayscale_t median_brightness = 0;
    double cumul = 0.0;
    for (size_t i = 0; i < max_brightness; i++) {
      cumul += brightness_histogram[i];
      if (cumul > .5) {
        median_brightness = (grayscale_t) ((i > 0) ? i - 1 : 0);
        break;
      }
    }
    binaryScalingByCap(image, median_brightness);
    return true;
  }

  bool Inversion::transform(GrayscaleImage &image) {
    std::for_each(image.begin(), image.end(), [](auto &e) { e = max_brightness - e; });
    return true;
  }
}   // namespace image::transform
