#include "Image.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <dirent.h>
#include <unistd.h>

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "stb_image_write.h"

namespace image {

Color::Color(color_t r, color_t g, color_t b) : r(r), g(g), b(b) {}

void Color::print() const {
    std::cout << "[" << (unsigned int)this->r << ";" << (unsigned int)this->g
              << ";" << (unsigned int)this->b << "]";
}

Image::Image(size_t width, size_t height) : width(width), height(height) {
    if (width == 0 || height == 0) return;
    colors = std::make_unique<Color[]>(width * height);
}

Image::Image(size_t width, size_t height, std::unique_ptr<Color[]> &&ptr)
    : width(width), height(height) {
    if (width == 0 || height == 0) return;
    assign(std::move(ptr));
}

Image::Image(Image const &other) { *this = other; }

Image::Image(Image &&other) { *this = std::move(other); }

Image &Image::operator=(Image const &other) {
    if (this == &other) return *this;

    size_t size = other.getDimension();

    if (getDimension() != size) {
        colors = nullptr;
    }

    width = other.width;
    height = other.height;

    if (other.colors) {
        if (not colors) colors = std::make_unique<Color[]>(size);
        std::memcpy(colors.get(), other.colors.get(), sizeof(Color) * size);
    }

    return *this;
}

Image &Image::operator=(Image &&other) {
    if (this == &other) return *this;

    colors = std::move(other.colors);
    width = other.width;
    height = other.height;
    other.width = 0;
    other.height = 0;

    return *this;
}

Color *Image::begin() { return getData(); }
Color *Image::end() { return getData() + getDimension(); }

const Color *Image::begin() const { return getData(); }
const Color *Image::end() const { return getData() + getDimension(); }

const Color *Image::getData() const { return colors.get(); }
Color *Image::getData() { return colors.get(); }

void Image::print() const {
    std::cout << "[" << std::endl;
    for (unsigned l = 0; l < height; l++) {
        for (unsigned c = 0; c < width; c++) {
            colors[l * width + c].print();
            std::cout << " ; ";
        }
        std::cout << std::endl;
    }
    std::cout << "]" << std::endl;
}

color_t Image::getMaxColor() const {
    color_t max = 0;
    for (Color const &each : *this) {
        color_t local_max = std::max(each.g, std::max(each.r, each.b));
        max = std::max(local_max, max);
    }
    return max;
}

size_t Image::getWidth() const { return width; }
size_t Image::getHeight() const { return height; }
size_t Image::getDimension() const { return getHeight() * getWidth(); }

double Image::difference(const Image &other) const {
    assert(other.width == width);
    assert(other.height == height);
    // assert(other.colors.size() == colors.size());

    double diff = 0.0;
    for (size_t i = 0; i < getDimension(); i++) {
        size_t this_sum = colors[i].r + colors[i].g + colors[i].b;
        size_t other_sum =
            other.colors[i].r + other.colors[i].g + other.colors[i].b;
        diff += ((double)(std::max(this_sum, other_sum) -
                          std::min(this_sum, other_sum))) /
                possible_brightness;
    }
    return diff;
}

namespace {

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
    for (auto file : files) std::cout << file << "| ";
    std::cout << std::endl;
}

char _colorValueToAscii(color_t value) {
    return static_cast<char>(((int)value));
}

void _showImageInBrowser(std::string const filename) {
    if (system(nullptr) != -1) {
        char cmd[256];
        std::string pngImage = std::string(filename);
        pngImage.replace(pngImage.find_last_of('.') + 1, 3, std::string("png"));
        std::cout << pngImage << std::endl;
        sprintf(cmd, "convert %s %s", filename.data(), pngImage.data());
        std::cout << cmd << std::endl;
        system(cmd);
        sprintf(cmd, "firefox --new-tab -url `pwd`/%s", pngImage.data());
        std::cout << cmd << std::endl;
        system(cmd);
        sleep(2);
        sprintf(cmd, "rm %s", pngImage.data());
        std::cout << cmd << std::endl;
        system(cmd);
    }
}
}  // namespace

Image ImageLoader::createRandomImage() {
    Image res((rand() % 1080) + 1, (rand() % 1080) + 1);
    Color *raw_array = res.getData();
    for (size_t i = 0; i < res.getDimension(); i++) {
        raw_array[i] =
            (Color){rand() % nb_colors, rand() % nb_colors, rand() % nb_colors};
    }
    return res;
}

/**
 * @param filename: ONLY PPM FILES FOR NOW
 */
Image ImageLoader::load(std::string const filename) {
    std::ifstream fp;

    //_listDirectories(".");

    fp.open(filename);
    if (!fp.is_open()) {
        std::cerr << "<!> ImageLoader::load(" << filename
                  << ") -> cannot open file!" << std::endl;
        exit(-1);
    }
    fp.seekg(3);
    unsigned width, height, max_color = 0;
    fp >> width;
    assert(width > 0);
    fp.get();
    fp >> height;
    assert(height > 0);
    fp.get();
    fp >> max_color;
    assert(max_color == 255);
    fp.get();
    std::cout << "width: " << width << "; height: " << height
              << "; max_color: " << max_color << std::endl;

    int r, g, b = 0;
    char current;

    Image img(width, height);
    size_t i = 0;
    for (Color &each : img) {
        current = fp.get();
        // std::cout << "current = " << current << std::endl;
        r = current;
        current = fp.get();
        // std::cout << "current = " << current << std::endl;
        g = current;
        current = fp.get();
        // std::cout << "current = " << current << std::endl;
        b = current;
        // std::cout << "r: " << r << "; g: " << g << "; b: " << b <<
        // std::endl;
        each = {(color_t)r, (color_t)g, (color_t)b};
        // col.print();
        i++;
    }
    std::cout << "lecture OK" << std::endl;
    // img.print();
    fp.close();
    return img;
}

/**
 * @param filename: ONLY PPM FILES FOR NOW
 */
void ImageLoader::save(std::string const filename, Image const &image) {
    std::ofstream fp;

    //_listDirectories(".");

    fp.open(filename, std::ios_base::out | std::ios_base::binary);
    if (!fp.is_open()) {
        std::cerr << "<!> ImageLoader::save(" << filename
                  << ") -> cannot open file!" << std::endl;
        exit(-1);
    }

    // image.print();

    fp << "P6\n"
       << image.getWidth() << ' ' << image.getHeight() << '\n'
       << 255 << std::endl;
    for (Color current : image)
        fp << _colorValueToAscii(current.r) << _colorValueToAscii(current.g)
           << _colorValueToAscii(current.b);
    // fp << current.r << current.g << current.b;

    std::cout << "ecriture OK" << std::endl;
    fp.close();
    _showImageInBrowser(filename);
}
/**
 * @param filename: any image file supported by stb.
 */
Image ImageLoader::load_stb(const char *filename) {
    int width, height, channels;
    unsigned char *imgData = stbi_load(filename, &width, &height, &channels, 3);
    if (imgData == NULL) {
        std::cout << "Error, cannot open \"" << filename << "\"." << std::endl;
        width = 0;
        height = 0;
        Image img((size_t)width, (size_t)height);
        return img;
    }

    std::unique_ptr<Color[]> ptr(reinterpret_cast<Color *>(imgData));
    std::cout << ptr.get() << std::endl;
    Image img((size_t)width, (size_t)height, std::move(ptr));

    return img;
}

}  // namespace image
