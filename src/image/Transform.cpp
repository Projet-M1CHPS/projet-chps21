#include "Transform.hpp"

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <math.h>
#include <numeric>
#include <stdexcept>

namespace image::transform {

  Crop::Crop(size_t width, size_t height, size_t orig_x, size_t orig_y)
      : width(width), height(height), orig_x(orig_x), orig_y(orig_y){};

  Resize::Resize(size_t width, size_t height) : width(width), height(height){};
  Restriction::Restriction(size_t desired_step) : desired_step(desired_step){};

  namespace filtering_subfunctions {

    float gaussian2D(float x_distance, float y_distance) {
      float sigma = 1.84f;
      return expf(-(x_distance * x_distance + y_distance * y_distance) / (2 * sigma * sigma)) *
             (1 / (2 * M_PI * sigma * sigma));
    }

    std::vector<float> gaussianFilter(size_t dimension) {
      std::vector<float> filter(dimension * dimension);
      int half_d = std::floor(dimension / 2.0f);
      for (int y = 0; y < dimension; y++) {
        for (int x = 0; x <= dimension; x++) {
          float x_dist_from_center = fabs(x - half_d);
          float y_dist_from_center = fabs(y - half_d);
          filter[y * dimension + x] = gaussian2D(x_dist_from_center, y_dist_from_center);
        }
      }
      return filter;
    }

    float applyFilterAt(const GrayscaleImage &img, size_t orig_x, size_t orig_y,
                        std::vector<float> const &filter, size_t dimension) {
      float sum = .0f;
      size_t half_dimension = std::floor(dimension / 2.0);
      size_t filter_idx = 0;
      for (size_t y = 0; y < dimension; y++)
        for (size_t x = 0; x < dimension; x++)
          sum += (float) img(orig_x + x - half_dimension, orig_y + y - half_dimension) *
                 filter[filter_idx++];
      return sum;
    }
  }   // namespace filtering_subfunctions


  float _get2DVectorNorm(size_t orig_x, size_t orig_y, size_t dest_x, size_t dest_y) {
    size_t xdist = dest_x - orig_x;
    size_t ydist = dest_y - orig_y;
    return sqrtf(xdist * xdist + ydist * ydist);
  }


  /**
   * @brief Sub-function for resizing (stable on both up-scaling and down-scaling, but not really
   * efficient on up-scaling)
   *
   * @param factors Pair: (source.width-1)/(dest.width-1) ; (source.height-1)/(dest.height-1)
   * @param source source image
   * @param dest destination image
   */
  void downscaling(std::pair<double, double> factors, GrayscaleImage const &source,
                   GrayscaleImage &dest) {
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
        float top_left_norm = _get2DVectorNorm(x, y, orig_left_x, orig_up_y);
        float top_right_norm = _get2DVectorNorm(x, y, orig_right_x, orig_up_y);
        float bottom_left_norm = _get2DVectorNorm(x, y, orig_left_x, orig_down_y);
        float bottom_right_norm = _get2DVectorNorm(x, y, orig_right_x, orig_down_y);

        float cumul_norm = 2 * (top_left_norm + bottom_right_norm);

        grayscale_t mean =
                (grayscale_t) std::round((source(orig_left_x, orig_up_y) * top_left_norm +
                                          source(orig_left_x, orig_down_y) * bottom_left_norm +
                                          source(orig_right_x, orig_up_y) * top_right_norm +
                                          source(orig_right_x, orig_down_y) * bottom_right_norm) /
                                         (cumul_norm));
        dest(x, y) = mean;
      }
    }
  }

  bool Crop::transform(GrayscaleImage &image) {
    if (orig_x < 0 || orig_y < 0 || width <= 0 || height <= 0) {
      throw std::invalid_argument(
              "Crop can not be applied on this image: new origin needs to be greater or equal than "
              "(0,0) and new dimensions needs to be strictly greater than (0,0)");
    } else if (orig_x + width > image.getWidth()) {
      throw std::invalid_argument("Crop::transform: orig_x + width > image.getWidth()");
    } else if (orig_y + height > image.getHeight()) {
      throw std::invalid_argument("Crop::transform: orig_y + height > image.getHeight()");
    }
    const image::GrayscaleImage source =
            image;   // In order to modify the image ref, we first need to copy it.
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
      throw std::invalid_argument("Resize can not be applied on this image: new dimensions needs "
                                  "to be strictly greater than (0,0)");
    } else if (height == image.getHeight() &&
               width == image.getWidth())   // Case where we don't need to resize the image.
      return true;
    const image::GrayscaleImage source =
            image;   // In order to modify the image ref, we first need to copy it.
    image.setSize(width, height);
    std::pair<double, double> factors((double) (source.getWidth() - 1) / (width - 1),
                                      (double) (source.getHeight() - 1) / (height - 1));
    // Works for both up & down scaling.
    // A upscaling(...) function will be defined soon to enhance that particular case.
    downscaling(factors, source, image);
    return true;
  }

  bool Restriction::transform(GrayscaleImage &image) {
    unsigned compare_step = desired_step + 1;
    std::for_each(image.begin(), image.end(), [compare_step](auto &e) {
      unsigned modulo_value = (e % compare_step);
      if (modulo_value != 0) { e -= modulo_value; }
    });
    return true;
  }

  // TODO: Improve performance, really slow.
  bool Filter::transform(GrayscaleImage &image) {
    size_t dimension = 11;   // Needs to be odd (2n+1)
    size_t half_d = std::floor(dimension / 2.0);
    std::vector<float> filter = filtering_subfunctions::gaussianFilter(dimension);

    for (size_t y = half_d; y < image.getHeight() - half_d; y++)
      for (size_t x = half_d; x < image.getWidth() - half_d; x++)
        image(x, y) =
                (grayscale_t) filtering_subfunctions::applyFilterAt(image, x, y, filter, dimension);
    return true;
  }

  bool Edges::transform(GrayscaleImage &image) {
    size_t cap = 50;   // over 1020
    int sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    std::vector<size_t> edges;
    for (size_t y = 1; y < image.getHeight() - 1; y++) {
      for (size_t x = 1; x < image.getWidth() - 1; x++) {
        int current[9] = {image(x - 1, y - 1), image(x - 1, y), image(x - 1, y + 1),
                          image(x, y - 1),     image(x, y),     image(x, y + 1),
                          image(x + 1, y - 1), image(x + 1, y), image(x + 1, y + 1)};

        int gx = 0;
        int gy = 0;
        for (int i = 0; i < 9; i++) {
          gx += sobel_x[i] * current[i];
          gy += sobel_y[i] * current[i];
        }

        if (std::abs(gx) >= cap || std::abs(gy) >= cap) edges.push_back(y * image.getHeight() + x);
      }
    }

    size_t cur_edge = 0;
    for (size_t i = image.getWidth() + 1; i < image.getSize() - image.getWidth(); i++) {
      if (edges[cur_edge] == i) {
        image.getData()[i] = 255U;
        cur_edge += 1;
      } else
        image.getData()[i] = 0U;
    }
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
    std::for_each(image.begin(), image.end(), [cap](auto &e) { e = e <= cap ? 0U : 255U; });
  }

  bool BinaryScale::transform(GrayscaleImage &image) {
    float half_brightness = max_brightness / 2.0;
    binaryScalingByCap(image, half_brightness);
    return true;
  }

  bool BinaryScaleByMedian::transform(GrayscaleImage &image) {
    std::vector<double> brightness_histogram = image.createRatioHistogram();
    grayscale_t median_brightness = 254;
    double cumul = 0.0;
    for (size_t i = 0; i < max_brightness; i++) {
      cumul += brightness_histogram[i];
      if (cumul >= .5) {
        median_brightness = (grayscale_t) i;
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
