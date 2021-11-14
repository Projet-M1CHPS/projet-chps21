#pragma once

#include <vector>
#include <memory>

//using color_t = unsigned char;
using grayscale_color = unsigned char;

constexpr unsigned possible_brightness = 255 * 3;
constexpr unsigned nb_colors = 256;

namespace image {

/*struct RGBColor {
    RGBColor() = default;
    RGBColor(color_t r, color_t g, color_t b);
    size_t getBrightness();
    void print() const;
    color_t r, g, b;
};

static_assert(sizeof(RGBColor) == sizeof(unsigned char) * 3, "Invalid color size");
*/

class Image {
    public:
    // constructors
        // init image with a size
        Image(size_t width = 0, size_t height = 0);
        // init and fill image with an array of grayscale
        Image(size_t width, size_t height, std::unique_ptr<grayscale_color[]>&& ptr);
        // copy another already existing image
        Image(Image const& other); 
        Image(Image&& other);
    // destructor
    // operators
        Image& operator=(Image const& other);
        Image& operator=(Image&& other);
    // color manipulation
        //assign a new array of grayscale to the class
        void assign(std::unique_ptr<grayscale_color[]>&& data);
    // getters
        // get grayscale data
        //// pixel
        const grayscale_color getPixel(unsigned int x) const;
        const grayscale_color getPixel(unsigned int x, unsigned int y) const;
        //// whole array
        grayscale_color* getData();
        const grayscale_color* getData() const;
        // get sizes
        size_t getWidth() const;
        size_t getHeight() const;
        size_t getDimension() const; // returns width * height
        // get the first pixel of the image
        grayscale_color* begin();
        const grayscale_color* begin() const;
        // get the last pixel of the image
        grayscale_color* end();
        const grayscale_color* end() const;
    // other
        void print() const;
    private:
    std::unique_ptr<grayscale_color[]> pixel_data;
    size_t width, height, dimension;
};

// Can be replaced by factory methods inside the Image class
// Anyhow, we must be able to easily load a whole directory (recursively or not)
// and handle different image formats (atleast ppm)
class ImageLoader {
   public:
    static image::Image createRandomImage();
    static image::Image load_stb(const char* filename);
    static void save_png_stb(const char* filename, const Image& image);
    static std::vector<image::Image> loadDirectory(std::string const filename);

   private:
};
}  // namespace image