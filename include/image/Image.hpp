#pragma once

#include "Matrix.hpp"
#include <filesystem>
#include <memory>
#include <vector>

namespace image {
  using grayscale_t = unsigned char;

  constexpr unsigned max_brightness = 255;
  constexpr unsigned nb_colors = 256;

  /** @brief Stores an image in grayscale format
   *
   * Provides access to the underlying array for transformations
   *
   */
  class GrayscaleImage {
  public:
    /**
     * @brief Construct a new GrayscaleImage object
     *
     * @param width Default to 0
     * @param height Default to 0
     */
    explicit GrayscaleImage(size_t width = 0, size_t height = 0);

    /**
     * @brief Construct a new GrayscaleImage object
     *
     * @param width width of the image
     * @param height height of the image
     * @param ptr Ptr to the raw data
     * The image assume the ownership of the ptr
     */
    GrayscaleImage(size_t width, size_t height, std::unique_ptr<grayscale_t[]> &&ptr);

    GrayscaleImage(GrayscaleImage const &other);
    GrayscaleImage &operator=(GrayscaleImage const &other);

    GrayscaleImage(GrayscaleImage &&other) noexcept;
    GrayscaleImage &operator=(GrayscaleImage &&other) noexcept;

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
    [[nodiscard]] grayscale_t getPixel(unsigned int x) const;

    /**
     * @brief Returns the pixel at coordinate (x, y)
     *
     * Throws on out of bound access
     *
     * @param x
     * @return grayscale_t
     */
    [[nodiscard]] grayscale_t getPixel(unsigned int x, unsigned int y) const;


    /**
     * @brief Compute the difference of the pixels between two images.
     * If images do not have the same dimensions, every not comparable pixel will be computed as a
     * 0% difference.
     *
     * @param other other image
     * @return difference in range [0.0, 1.0]
     */
    [[nodiscard]] double getDifference(GrayscaleImage const &other) const;


    /**
     * @brief Returns the underlying raw ptr
     */
    grayscale_t *getData();

    /**
     * @return [number of pixels with brightness = 0; ... = 1; ... =
     * max_brightness]
     */
    const std::vector<size_t> getHistogram() const;

    /**
     * @return [ratio of pixels with brightness[0.0; 1.0] = 0; ... = 1; ... =
     * max_brightness].
     * Cumulated value for this vector should be 1.0.
     */
    const std::vector<double> createRatioHistogram() const;

    grayscale_t &operator()(size_t x, size_t y) { return getData()[x + y * width]; }

    grayscale_t operator()(size_t x, size_t y) const { return getData()[x + y * width]; }

    /**
     * @brief Returns the underlying raw ptr
     */
    [[nodiscard]] const grayscale_t *getData() const;

    [[nodiscard]] size_t getWidth() const { return width; }

    [[nodiscard]] size_t getHeight() const { return height; }

    /**
     * @brief Returns the dimensions of the image as a pair
     *
     * @return std::pair<width, height>
     */
    [[nodiscard]] std::pair<size_t, size_t> getDimension() const {
      return std::make_pair(width, height);
    }

    /**
     * @brief Getter for the image area
     *
     * @return Number of pixel in the image
     */
    [[nodiscard]] size_t getSize() const { return height * width; }

    /**
     * @brief Change the images dimensions properties (width, height) and re-allocate the pixels
     * array accordingly.
     */
    void setSize(size_t new_width, size_t new_height) {
      if (new_width <= 0) throw std::invalid_argument("setSize needs a new_width > 0");
      else if (new_height <= 0)
        throw std::invalid_argument("setSize needs a new_height > 0");
      std::tie(width, height) = {new_width, new_height};
      pixel_data = std::make_unique<grayscale_t[]>(width * height);
    }

    [[nodiscard]] grayscale_t *begin();
    [[nodiscard]] const grayscale_t *begin() const;

    [[nodiscard]] grayscale_t *end();
    [[nodiscard]] const grayscale_t *end() const;

  private:
    std::unique_ptr<grayscale_t[]> pixel_data;
    size_t width, height;
  };

  /** @brief Helper class for image generation and serialization
   *
   */
  class ImageSerializer {
  public:
    /**
     * @brief Create and return a random generated image
     *
     * @param width: DEFAULT = random number in range [124,512]
     * @param height: DEFAULT = random number in range [124,512]
     */
    static image::GrayscaleImage createRandomNoiseImage(size_t width, size_t height);

    /**
     * @brief Create and return a random generated image.
     *
     * @param width: DEFAULT = random number in range [124,512]
     * @param height: DEFAULT = random number in range [124,512]
     */
    static image::GrayscaleImage createRandomNoiseImage();

    /** @brief Load an image in any supported format from the given path
     *
     * @param filename
     * @return
     */
    static image::GrayscaleImage load(std::filesystem::path const &filename);

    /** @brief Saves an image as a PNG to the given path
     *
     * @param filename
     * @param image
     */
    static void save(std::string const &filename, const GrayscaleImage &image);

    /** @brief Returns a vector containing all the png images located in the given directory
     *
     * @param directory_path
     */
    static std::vector<image::GrayscaleImage>
    loadDirectory(std::filesystem::path const &directory_path);

    /**
     * @brief Get basic informations about a file without loading it
     *
     * @param path Image path
     * @return (width, height, channels)
     */
    static std::tuple<int, int, int> loadInfo(std::filesystem::path const &path);

  private:
  };
}   // namespace image