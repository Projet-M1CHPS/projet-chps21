#pragma once

#include <array>
#include <vector>
#include <memory>

#define MAX_COLOR 256
#define POSSIBLE_BRIGHTNESSES 255*3

using color_t = unsigned char;

namespace image {

    class Color {
        public:
            Color(color_t r, color_t g, color_t b);
            void print();
            color_t r, g, b;
        private:
    };
    // TODO: implement me
    // We must be able to access the raw data
    // To feed it to the neural network
    class Image {
        public :
            Image();
            Image(unsigned width, unsigned height, std::vector<Color> colors); // RGB Vector
            void print();
            unsigned width, height;
            std::vector<Color> colors;
        private :

    };

    // Can be replaced by factory methods inside the Image class
    // Anyhow, we must be able to easily load a whole directory (recursively or not)
    // and handle different image formats (atleast ppm)
    class ImageLoader {
        public :
            static image::Image load(std::string const filename);
            static void save(std::string const filename, const Image &image);
            static std::vector<image::Image> loadDirectory(std::string const filename);
        private :
    };
}