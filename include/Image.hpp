#pragma once

#include <vector>
#include <memory>

using color_t = unsigned char;

constexpr unsigned possible_brightness = 255 * 3;
constexpr unsigned nb_colors = 256;

namespace image {

struct RGBColor {
    RGBColor() = default;
    RGBColor(color_t r, color_t g, color_t b);
    size_t getBrightness();
    void print() const;
    color_t r, g, b;
};

static_assert(sizeof(RGBColor) == sizeof(unsigned char) * 3, "Invalid color size");

// TODO: implement me
// We must be able to access the raw data
// To feed it to the neural network
class Image {
   public:
    Image(size_t width = 0, size_t height = 0);  // RGB Vector
    Image(size_t width, size_t height,
          std::unique_ptr<RGBColor[]>&& ptr);  // RGB Vector

    Image(Image const& other);
    Image(Image&& other);

    Image& operator=(Image const& other);
    Image& operator=(Image&& other);

    RGBColor* getData();
    const RGBColor* getData() const;

    void assign(std::unique_ptr<RGBColor[]>&& data) { colors = std::move(data); }

    size_t getWidth() const;
    size_t getHeight() const;
    size_t getDimension() const; // returns width * height

    RGBColor* begin();
    const RGBColor* begin() const;

    RGBColor* end();
    const RGBColor* end() const;

    const RGBColor* cbegin() const;
    const RGBColor* cend() const;

    void print() const;
    color_t getMaxColor() const;

    /** @param other with the exact same dimension as this image
     *  @return the difference in percentage (%) between the two images pixels.
     */
    double difference(const Image& other) const;

   private:
    std::unique_ptr<RGBColor[]> colors;
    size_t width, height;
};

// Can be replaced by factory methods inside the Image class
// Anyhow, we must be able to easily load a whole directory (recursively or not)
// and handle different image formats (atleast ppm)
class ImageLoader {
   public:
    static image::Image createRandomImage();
    static image::Image load(std::string const filename);
    static image::Image load_stb(const char* filename);
    static void save(std::string const filename, const Image& image);
    static void save_png_stb(const char* filename, const Image& image);
    static std::vector<image::Image> loadDirectory(std::string const filename);

   private:
};
}  // namespace image