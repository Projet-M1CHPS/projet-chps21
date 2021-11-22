#pragma once

#include <memory>
#include <vector>

namespace image {

  using grayscale_t = unsigned char;

  constexpr unsigned max_brightness = 255;
  constexpr unsigned nb_colors = 256;

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
    GrayscaleImage(size_t width, size_t height,
                   std::unique_ptr<grayscale_t[]> &&ptr);

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
     * If images do not have the same dimensions, every not comparable pixel will be computed as a 0% difference.
     *
     * @param other other image
     * @return difference in range [0.0, 1.0]
     */
    [[nodiscard]] double getDifference(GrayscaleImage const &other) const;


    /**
     * @brief Returns the underlying raw ptr
     *
     * @return grayscale_t*
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

    grayscale_t &operator()(size_t x, size_t y) {
      return getData()[x + y * width];
    }

    grayscale_t operator()(size_t x, size_t y) const {
      return getData()[x + y * width];
    }

    /**
     * @brief Returns the underlying raw ptr
     *
     * @return const grayscale_t*
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
     * @brief Returns the number of pixel in the image
     * Equal to width * height
     *
     * @return size_t
     */
    [[nodiscard]] size_t getSize() const { return height * width; }

    void setSize(size_t new_width, size_t new_height) {
      width = new_width;
      height = new_height;
      pixel_data.reset();
      pixel_data = std::make_unique<grayscale_t[]>(width * height);
    }

    [[nodiscard]] grayscale_t *begin();
    [[nodiscard]] const grayscale_t *begin() const;

    [[nodiscard]] grayscale_t *end();
    [[nodiscard]] const grayscale_t *end() const;

  private:
    std::unique_ptr<grayscale_t[]> pixel_data;
    size_t width{}, height{};
  };

  /**
   * @brief Helper class for image generation and serialization
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
    static image::GrayscaleImage load(std::string const &filename);

    /** @brief Saves an image as a PNG to the given path
     *
     * @param filename
     * @param image
     */
    static void save(std::string const &filename, const GrayscaleImage &image);

    static std::vector<image::GrayscaleImage>
    loadDirectory(std::string const &filename);

  private:
  };
}   // namespace image