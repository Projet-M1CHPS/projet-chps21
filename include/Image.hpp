#pragma once

#include <memory>
#include <vector>

using grayscale_t = unsigned char;

constexpr unsigned max_brightness = 255;
constexpr unsigned nb_colors = 256;

namespace image {

  class GrayscaleImage {
  public:
    /**
     * @brief Construct a new GrayscaleImage object
     *
     * @param width Default to 0
     * @param height Default to 0
     */
    GrayscaleImage(size_t width = 0, size_t height = 0);

    /**
     * @brief Construct a new GrayscaleImage object
     *
     * @param width width of the image
     * @param height height of the image
     * @param ptr Ptr to the raw data
     * The image assume the ownership of the ptr
     */
    GrayscaleImage(size_t width, size_t height,
                   std::unique_ptr<grayscale_t[]> &&ptr);

    GrayscaleImage(GrayscaleImage const &other);
    GrayscaleImage &operator=(GrayscaleImage const &other);

    GrayscaleImage(GrayscaleImage &&other);
    GrayscaleImage &operator=(GrayscaleImage &&other);

    ~GrayscaleImage() = default;

    /**
     * @brief Reassign the image raw pointer
     * The image assumes ownership of the ptr
     *
     * @param data
     */
    void assign(std::unique_ptr<grayscale_t[]> &&data);

    /**
     * @brief Returns the nth pixel
     *
     * Throws on out of bound access
     *
     * @param x
     * @return grayscale_t
     */
    grayscale_t getPixel(unsigned int x) const;

    /**
     * @brief Returns the pixel at coordinate (x, y)
     *
     * Throws on out of bound access
     *
     * @param x
     * @return grayscale_t
     */
    grayscale_t getPixel(unsigned int x, unsigned int y) const;


    /**
     * @brief Compute the difference of the pixels between two images.
     * If images do not have the same dimensions, every not comparable pixel will be computed as a 0% difference.
     *
     * @param other another image
     * @return difference in range [0.0, 1.0]
     */
    double getDifference(GrayscaleImage const &other) const;


    /**
     * @brief Returns the underlying raw ptr
     *
     * @return grayscale_t*
     */
    grayscale_t *getData();

    /**
     * @brief Returns the underlying raw ptr
     *
     * @return const grayscale_t*
     */
    const grayscale_t *getData() const;

    size_t getWidth() const { return width; }

    size_t getHeight() const { return height; }

    /**
     * @brief Returns the dimensions of the image as a pair
     *
     * @return std::pair<size_t, size_t>
     */
    std::pair<size_t, size_t> getDimension() const {
      return std::make_pair(width, height);
    }

    /**
     * @brief Returns the number of pixel in the image
     * Equal to width * height
     *
     * @return size_t
     */
    size_t getSize() const { return height * width; }

    grayscale_t *begin();
    const grayscale_t *begin() const;

    grayscale_t *end();
    const grayscale_t *end() const;

  private:
    std::unique_ptr<grayscale_t[]> pixel_data;
    size_t width, height;
  };

  /**
   * @brief Helper class for image generation and serialization
   *
   */
  class ImageLoader {
  public:
    /**
     * @brief Create and return a random generated image
     * @param width image width, default is random in range [1:1080]
     * @param height image height, default is random in range [1:1080]
     */
    static image::GrayscaleImage createRandomNoiseImage(size_t width, size_t height);
    /**
     * @brief Create and return a random generated image
     * @param width image width, default is random in range [1:1080]
     * @param height image height, default is random in range [1:1080]
     */
    static image::GrayscaleImage createRandomNoiseImage();

    static image::GrayscaleImage load(std::string const &filename);
    static void save(std::string const &filename, const GrayscaleImage &image);
    static std::vector<image::GrayscaleImage>
    loadDirectory(std::string const &filename);

  private:
  };
}   // namespace image