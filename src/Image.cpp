#include "Image.hpp"

#include <dirent.h>
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


Image::Image(size_t width, size_t height) : width(width), height(height), dimension(width*height) {
    if (width == 0 || height == 0) return;
    pixel_data = std::make_unique<grayscale_color[]>(width * height);
}

Image::Image(size_t width, size_t height, std::unique_ptr<grayscale_color[]> &&ptr)
    : width(width), height(height), dimension(width*height) {
    if (width == 0 || height == 0) return;
    assign(std::move(ptr));
}

Image::Image(Image const &other) { *this = other; }

Image::Image(Image &&other) { *this = std::move(other); }

Image &Image::operator=(Image const &other) {
    if (this == &other) return *this;

    if (getDimension() != other.getDimension()) {
        this->pixel_data = nullptr;
    }

    this->width = other.getWidth();
    this->height = other.getHeight();
    this->dimension = other.getDimension();

    if (other.pixel_data) {
        if (not this->pixel_data) this->pixel_data = std::make_unique<grayscale_color[]>(this->dimension);
        std::memcpy(this->pixel_data.get(), other.pixel_data.get(), sizeof(grayscale_color) * this->dimension);
    }

    return *this;
}

Image &Image::operator=(Image &&other) {
    if (this == &other) return *this;

    this->pixel_data = std::move(other.pixel_data);
    width = other.getWidth();
    height = other.getHeight();
    this->dimension = other.getDimension();
    other.width = 0;
    other.height = 0;
    other.dimension = 0;

    return *this;
}

void Image::assign(std::unique_ptr<grayscale_color[]>&& data) {
    pixel_data = std::move(data);
}

const grayscale_color Image::getPixel(unsigned int x) const{
    if (x < this->getDimension()) {
        return pixel_data.get()[x];
    }
    else {
        return 0;
    }
}

const grayscale_color Image::getPixel(unsigned int x, unsigned int y) const{
    if (x < this->getWidth() && y < this->getHeight()) {
        return pixel_data.get()[this->getWidth() * x + y];
    }
    else {
        return 0;
    }
}

const grayscale_color *Image::getData() const { return pixel_data.get(); }
grayscale_color *Image::getData() { return pixel_data.get(); }

size_t Image::getWidth() const { return this->width; }
size_t Image::getHeight() const { return this->height; }
size_t Image::getDimension() const { return this->dimension; }


grayscale_color *Image::begin() { return getData(); }
grayscale_color *Image::end() { return getData() + getDimension(); }

const grayscale_color *Image::begin() const { return getData(); }
const grayscale_color *Image::end() const { return getData() + getDimension(); }

void Image::print() const {
    for (unsigned x = 0; x < this->width; x++) {
        for (unsigned y = 0; y < this-> height; y++) {
            std::cout << "[" << x << "][" << y << "] = " << this->getPixel(x,y) << std::endl;
        }
    }
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
    grayscale_color *raw_array = res.getData();
    for (size_t i = 0; i < res.getDimension(); i++) {
        raw_array[i] = (grayscale_color)(rand() % nb_colors);
    }
    return res;
}

/**
 * @param filename: any image file supported by stb.
 */
Image ImageLoader::load_stb(const char *filename) {
    int width, height, channels;
    unsigned char *img_data = stbi_load(filename, &width, &height, &channels, 3);
    if (img_data == NULL) {
        std::cout << "Error, cannot open \"" << filename << "\"." << std::endl;
        width = 0;
        height = 0;
        Image img((size_t)width, (size_t)height);
        return img;
    }

    Image img((size_t)width, (size_t)height);
    unsigned int i = 0;
    for (grayscale_color &each : img) {
        each = (grayscale_color)img_data[i];
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
    unsigned char const *img_to_save = image.getData();
    grayscale_color const *color_array = image.getData();

    stbi_write_png(filename, image.getWidth(), image.getHeight(), 3, img_to_save, image.getWidth() * 3);
    delete img_to_save;
}

}  // namespace image
