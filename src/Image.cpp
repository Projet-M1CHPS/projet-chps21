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

RGBColor::RGBColor(color_t r, color_t g, color_t b) : r(r), g(g), b(b) {}

size_t RGBColor::getBrightness() { return r + g + b; }

void RGBColor::print() const {
    std::cout << "[" << (unsigned int)this->r << ";" << (unsigned int)this->g
              << ";" << (unsigned int)this->b << "]";
}

Image::Image(size_t width, size_t height) : width(width), height(height) {
    if (width == 0 || height == 0) return;
    colors = std::make_unique<RGBColor[]>(width * height);
}

Image::Image(size_t width, size_t height, std::unique_ptr<RGBColor[]> &&ptr)
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
        if (not colors) colors = std::make_unique<RGBColor[]>(size);
        std::memcpy(colors.get(), other.colors.get(), sizeof(RGBColor) * size);
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

RGBColor *Image::begin() { return getData(); }
RGBColor *Image::end() { return getData() + getDimension(); }

const RGBColor *Image::begin() const { return getData(); }
const RGBColor *Image::end() const { return getData() + getDimension(); }

const RGBColor *Image::getData() const { return colors.get(); }
RGBColor *Image::getData() { return colors.get(); }

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
    for (RGBColor const &each : *this) {
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
    RGBColor *raw_array = res.getData();
    for (size_t i = 0; i < res.getDimension(); i++) {
        raw_array[i] = (RGBColor){(color_t)(rand() % nb_colors),
                                  (color_t)(rand() % nb_colors),
                                  (color_t)(rand() % nb_colors)};
    }
    return res;
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

    Image img((size_t)width, (size_t)height);
    unsigned int i = 0;
    for (RGBColor &each : img) {
        each = {(color_t)imgData[i * 3], (color_t)imgData[i * 3 + 1], (color_t)imgData[i * 3 + 2]};
        i++;
    }

    //std::unique_ptr<RGBColor[]> ptr(reinterpret_cast<RGBColor *>(imgData));
    //std::cout << ptr.get() << std::endl;

    return img;
}

/**
 * @param filename: name of the png file generated (must end by .png).
 */
void ImageLoader::save_png_stb(const char* filename, Image const &image) {
    unsigned char *img_to_save = new unsigned char[image.getDimension() * 3];
    RGBColor const *color_array = image.getData();

    for (unsigned int i = 0; i < image.getDimension(); i++) {
                img_to_save[i * 3]     = color_array[i].r;
                img_to_save[i * 3 + 1] = color_array[i].g;
                img_to_save[i * 3 + 2] = color_array[i].b;
    }

    stbi_write_png(filename, image.getWidth(), image.getHeight(), 3, img_to_save, image.getWidth() * 3);
    delete img_to_save;
}

}  // namespace image
