#include "Image.hpp"

#include <dirent.h>
#include <numeric>
#include <unistd.h>

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace image {

  GrayscaleImage::GrayscaleImage(size_t width, size_t height)
      : width(width), height(height) {
    if (width == 0 || height == 0)
      return;
    pixel_data = std::make_unique<grayscale_t[]>(width * height);
  }

  GrayscaleImage::GrayscaleImage(size_t width, size_t height,
                                 std::unique_ptr<grayscale_t[]> &&ptr)
      : width(width), height(height) {
    if (width == 0 || height == 0)
      return;
    assign(std::move(ptr));
  }

  GrayscaleImage::GrayscaleImage(GrayscaleImage const &other) : width(0), height(0) { *this = other; }

  GrayscaleImage::GrayscaleImage(GrayscaleImage &&other) noexcept {
    *this = std::move(other);
  }

  GrayscaleImage &GrayscaleImage::operator=(GrayscaleImage const &other) {
    if (this == &other)
      return *this;

    // We may avoid a copy if the internal array is big enough to hold
    // the copy
    if (getSize() != other.getSize()) {
      pixel_data = nullptr;
    }

    width = other.width;
    height = other.height;
    if (other.pixel_data) {
      // If the internal array wasn't big enough or just not allocated
      if (not pixel_data) {
        pixel_data = std::make_unique<grayscale_t[]>(width * height);
      }
      std::memcpy(pixel_data.get(), other.pixel_data.get(),
                  sizeof(grayscale_t) * width * height);
    }

    return *this;
  }

  GrayscaleImage &GrayscaleImage::operator=(GrayscaleImage &&other) noexcept {
    if (this == &other)
      return *this;

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
    if (x > getSize())
      throw std::out_of_range("Out of range access in image");
    return pixel_data.get()[x];
  }

  grayscale_t GrayscaleImage::getPixel(unsigned int x, unsigned int y) const {
    if (x > width && y > getHeight())
      throw std::out_of_range("Out of range access in image");
    return pixel_data.get()[x];
  }

  double GrayscaleImage::getDifference(GrayscaleImage const &other) const {
    double diff = 0.0;
    const grayscale_t *self_data = getData(),
                      *other_data = other.getData();

    size_t stop = std::min(getSize(), other.getSize());

    for (size_t i = 0; i < stop; i++) {
      diff += std::fabs(self_data[i] - other_data[i]) / max_brightness;
    }
    return diff;
  }

  grayscale_t *GrayscaleImage::getData() { return pixel_data.get(); }
  const grayscale_t *GrayscaleImage::getData() const { return pixel_data.get(); }

  grayscale_t *GrayscaleImage::begin() { return getData(); }
  const grayscale_t *GrayscaleImage::begin() const { return getData(); }

  grayscale_t *GrayscaleImage::end() { return getData() + getSize(); }
  const grayscale_t *GrayscaleImage::end() const { return getData() + getSize(); }

  namespace {

    // FIXME: Remove this atrocity
    // For further Inquiry about the very nature of this function, please
    // refer to BENJAMIN LOZES
    void _listDirectories(char *path) {
      DIR *dir;
      struct dirent *diread;
      std::vector<char *> files;

      if ((dir = opendir(path)) != nullptr) {
        while ((diread = readdir(dir)) != nullptr)
          files.push_back(diread->d_name);
        closedir(dir);
      } else {
        perror("opendir");
        return;
      }
      for (auto file : files)
        std::cout << file << "| ";
      std::cout << std::endl;
    }

    // FIXME: Remove this atrocity
    // For further Inquiry about the very nature of this function, please
    // refer to BENJAMIN LOZES
    void _showImageInBrowser(std::string filename) {
      if (system(nullptr) != -1) {
        char cmd[256];
        filename.replace(filename.find_last_of('.') + 1, 3, "png");

        std::cout << filename << std::endl;
        sprintf(cmd, "convert %s %s", filename.data(), filename.data());
        std::cout << cmd << std::endl;

        system(cmd);
        sprintf(cmd, "firefox --new-tab -url `pwd`/%s", filename.data());
        std::cout << cmd << std::endl;
        system(cmd);

        sleep(2);
        sprintf(cmd, "rm %s", filename.data());

        std::cout << cmd << std::endl;
        system(cmd);
      }
    }
  }   // namespace

  GrayscaleImage ImageSerializer::createRandomNoiseImage(size_t width, size_t height) {
    GrayscaleImage res(width, height);
    grayscale_t *raw_array = res.getData();

    for (size_t i = 0; i < res.getSize(); i++) {
      raw_array[i] = (grayscale_t) (rand() % nb_colors);
    }
    return res;
  }

  GrayscaleImage ImageSerializer::createRandomNoiseImage() {
    return ImageSerializer::createRandomNoiseImage((size_t) (rand() % 1080) + 1, (size_t) (rand() % 1080) + 1);
  }

  /**
   * @param filename any image file supported by stb.
   */
  GrayscaleImage ImageSerializer::load(std::string const &filename) {
    int width, height, channels;
    unsigned char *img_data =
            stbi_load(filename.c_str(), &width, &height, &channels, 1);

    if (img_data == NULL)
      throw std::runtime_error("ImageSerializer::load: stbi_load failed");

    std::unique_ptr<grayscale_t[]> ptr(reinterpret_cast<grayscale_t *>(img_data));
    GrayscaleImage res(width, height, std::move(ptr));

    return res;
  }

  /**
   * @brief Saves a grayscale as a png file
   *
   * @param filename absolute or relative path
   * @param image
   */
  void ImageSerializer::save(std::string const &filename,
                             GrayscaleImage const &image) {
    stbi_write_png(filename.c_str(), image.getWidth(), image.getHeight(), 1,
                   image.getData(), image.getWidth());
  }

}   // namespace image
