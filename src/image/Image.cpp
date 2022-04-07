#include "Image.hpp"

#include <algorithm>
#include <dirent.h>
#include <filesystem>
#include <functional>
#include <numeric>
#include <unistd.h>

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace fs = std::filesystem;

namespace image {

  GrayscaleImage::GrayscaleImage(size_t width, size_t height) : width(width), height(height) {
    if (width == 0 || height == 0) return;
    pixel_data = std::make_unique<grayscale_t[]>(width * height);
  }

  GrayscaleImage::GrayscaleImage(size_t width, size_t height, std::unique_ptr<grayscale_t[]> &&ptr)
      : width(width), height(height) {
    if (width == 0 || height == 0) return;
    assign(std::move(ptr));
  }

  GrayscaleImage::GrayscaleImage(GrayscaleImage const &other) : width(0), height(0) {
    *this = other;
  }

  GrayscaleImage::GrayscaleImage(GrayscaleImage &&other) noexcept { *this = std::move(other); }

  GrayscaleImage &GrayscaleImage::operator=(GrayscaleImage const &other) {
    if (this == &other) return *this;

    // We may avoid a copy if the internal array is big enough to hold
    // the copy
    if (getSize() != other.getSize()) { pixel_data = nullptr; }

    width = other.width;
    height = other.height;
    if (other.pixel_data) {
      // If the internal array wasn't big enough or just not allocated
      if (not pixel_data) { pixel_data = std::make_unique<grayscale_t[]>(width * height); }
      std::memcpy(pixel_data.get(), other.pixel_data.get(), sizeof(grayscale_t) * width * height);
    }

    return *this;
  }

  GrayscaleImage &GrayscaleImage::operator=(GrayscaleImage &&other) noexcept {
    if (this == &other) return *this;

    pixel_data = std::move(other.pixel_data);
    width = other.width;
    height = other.height;

    other.width = 0;
    other.height = 0;

    return *this;
  }

  void GrayscaleImage::assign(std::unique_ptr<grayscale_t[]> &&data) {
    pixel_data = std::move(data);
  }

  grayscale_t GrayscaleImage::getPixel(unsigned int x) const {
    if (x > getSize()) throw std::out_of_range("Out of range access in image");
    return pixel_data.get()[x];
  }

  grayscale_t GrayscaleImage::getPixel(unsigned int x, unsigned int y) const {
    if (x > width && y > getHeight()) throw std::out_of_range("Out of range access in image");
    return pixel_data.get()[x + y * this->height];
  }

  double GrayscaleImage::getDifference(GrayscaleImage const &other) const {
    double diff = 0.0;
    size_t min_height = std::min(getHeight(), other.getHeight());
    size_t min_width = std::min(getWidth(), other.getWidth());

    for (size_t y = 0; y < min_height; y++) {
      size_t x = 0;
      diff += std::accumulate(begin() + getWidth() * y, begin() + getWidth() * y + min_width, 0.0,
                              [&x, y, other](auto a, auto b) {
                                return (double) std::fabs(b - other(x++, y)) / max_brightness;
                              });
    }
    return diff;
  }

  grayscale_t *GrayscaleImage::getData() { return pixel_data.get(); }
  const grayscale_t *GrayscaleImage::getData() const { return pixel_data.get(); }

  const std::vector<size_t> GrayscaleImage::getHistogram() const {
    std::vector<size_t> histogram(nb_colors);
    std::for_each(begin(), end(), [&histogram](auto e) { histogram[e] += 1U; });
    return histogram;
  }

  const std::vector<double> GrayscaleImage::createRatioHistogram() const {
    double increment_value = 1.0 / ((double) getSize());
    std::vector<double> histogram(nb_colors);
    std::for_each(begin(), end(),
                  [&histogram, increment_value](auto e) { histogram[e] += increment_value; });
    return histogram;
  }

  grayscale_t *GrayscaleImage::begin() { return getData(); }
  const grayscale_t *GrayscaleImage::begin() const { return getData(); }

  grayscale_t *GrayscaleImage::end() { return getData() + getSize(); }
  const grayscale_t *GrayscaleImage::end() const { return getData() + getSize(); }

  GrayscaleImage ImageSerializer::createRandomNoiseImage(size_t width, size_t height) {
    GrayscaleImage res(width, height);

    srand(time(nullptr));
    std::for_each(res.begin(), res.end(), [](auto &e) { e = (grayscale_t) (rand() % nb_colors); });
    return res;
  }

  GrayscaleImage ImageSerializer::createRandomNoiseImage() {
    srand(time(nullptr));
    return ImageSerializer::createRandomNoiseImage((size_t) (rand() % 389) + 124,
                                                   (size_t) (rand() % 389) + 124);
  }

  GrayscaleImage ImageSerializer::load(fs::path const &filename) {
    int width, height, channels;
    unsigned char *img_data = stbi_load(filename.c_str(), &width, &height, &channels, 1);

    if (img_data == nullptr) throw std::runtime_error("ImageSerializer::load: stbi_load failed");


    // We reallocate an array using a unique_ptr
    // Since stbi_load returns a pointer that must be freed by stbi_image_free
    // This is inefficient but atleast we can leave the memory management to the c++ standard
    std::unique_ptr<grayscale_t[]> ptr = std::make_unique<grayscale_t[]>(width * height);
    std::memcpy(ptr.get(), img_data, sizeof(grayscale_t) * width * height);
    stbi_image_free(img_data);

    GrayscaleImage res(width, height, std::move(ptr));

    return res;
  }

  void ImageSerializer::save(std::string const &filename, GrayscaleImage const &image) {
    stbi_write_png(filename.c_str(), image.getWidth(), image.getHeight(), 1, image.getData(),
                   image.getWidth());
  }


  void ImageSerializer::save(std::string const &filename, const math::clFMatrix &matrix,
                             float rescale) {
    ImageSerializer::save(filename, matrix.toFloatMatrix(), rescale);
  }

  void ImageSerializer::save(std::string const &filename, const math::FloatMatrix &image,
                             float rescale) {
    size_t width = image.getRows();
    size_t height = image.getCols();
    std::unique_ptr<grayscale_t[]> ptr = std::make_unique<grayscale_t[]>(width * height);

    for (size_t i = 0; i < width * height; i++) {
      ptr[i] = (grayscale_t) (image.getData()[i] * rescale);
    }
    GrayscaleImage res(width, height, std::move(ptr));
    ImageSerializer::save(filename, res);
  }


  std::vector<image::GrayscaleImage>
  ImageSerializer::loadDirectory(fs::path const &directory_path) {
    std::vector<image::GrayscaleImage> img_list;

    if (!fs::exists(directory_path)) {
      throw std::runtime_error(
              "Error: " + directory_path.string() +
              " doesnt exist. No images were loaded. Empty vector has been returned.\n");
      return img_list;
    }
    if (!fs::is_directory(directory_path)) {
      throw std::runtime_error(
              "Error: " + directory_path.string() +
              " is not a directory. No images were loaded. Empty vector has been returned.\n");
      return img_list;
    }
    if ((fs::status(directory_path).permissions() & fs::perms::others_read) == fs::perms::none) {
      throw std::runtime_error(
              "Error: " + directory_path.string() +
              " is not readable. No images were loaded. Empty vector has been returned.\n");
      return img_list;
    }

    for (const auto &file : fs::directory_iterator(directory_path))
      if (std::regex_match((std::string) file.path(), std::regex("(.*)(\\.png)"))) {
        img_list.push_back(image::ImageSerializer::load(file.path()));
      }
    if (img_list.size() == 0) {
      throw std::runtime_error("Warning, no images were loaded. The directory doesnt contain any "
                               "readable images.\n");
    }
    return img_list;
  }

  std::tuple<int, int, int> ImageSerializer::loadInfo(fs::path const &path) {
    int width, height, canals;
    stbi_info(path.c_str(), &width, &height, &canals);
    return {width, height, canals};
  }

}   // namespace image
