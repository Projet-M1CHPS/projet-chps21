#include "Image.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <dirent.h>
#include <unistd.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "stb_image_write.h"

namespace image {

Color::Color(color_t r, color_t g, color_t b) {
    this->r = r;
    this->g = g;
    this->b = b;
}

void Color::print() const {
    std::cout << "[" << (unsigned int)this->r << ";" << (unsigned int)this->g
              << ";" << (unsigned int)this->b << "]";
}

Image::Image() {
    this->width = 0;
    this->height = 0;
}

Image::Image(unsigned width, unsigned height, std::vector<Color> colors) {
    assert(colors.size() == width * height);
    this->width = width;
    this->height = height;
    this->colors = colors;
}

void Image::print() const {
    std::cout << "[" << std::endl;
    for (unsigned l = 0; l < this->width; l++) {
        for (unsigned c = 0; c < this->width; c++) {
            this->colors[l * this->width + c].print();
            std::cout << " ; ";
        }
        std::cout << std::endl;
    }
    std::cout << "]" << std::endl;
}

color_t Image::getMaxColor() const {
    color_t max = 0;
    for (Color each : this->colors) {
        if (each.r > max)
            max = each.r;
        else if (each.g > max)
            max = each.g;
        else if (each.b > max)
            max = each.b;
    }
    return max;
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
    return (char)static_cast<char>(((int)value) + '0');
}

void _showImageInBrowser(std::string const filename) {
    if (system(nullptr) != -1) {
        char cmd[256];
        std::string pngImage = std::string(filename);
        pngImage.replace(pngImage.find_last_of('.')+1, 3, std::string("png"));
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
    std::vector<Color> colors;
    int r, g, b = 0;
    char current;
    for (unsigned i = 0; i < width * height; i++) {
        current = fp.get();
        // std::cout << "current = " << current << std::endl;
        r = current - '0';
        current = fp.get();
        // std::cout << "current = " << current << std::endl;
        g = current - '0';
        current = fp.get();
        // std::cout << "current = " << current << std::endl;
        b = current - '0';
        // std::cout << "r: " << r << "; g: " << g << "; b: " << b <<
        // std::endl;
        Color col((color_t)r, (color_t)g, (color_t)b);
        // col.print();
        colors.push_back(col);
    }
    std::cout << "lecture OK" << std::endl;
    Image img(width, height, std::vector<Color>(colors));
    //img.print();
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

    //image.print();

    fp << "P6\n"
       << image.width << ' ' << image.height << '\n'
       << 255 << std::endl;
    for (Color current : image.colors)
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
        std::vector<Color> colors;
        Image img((unsigned int)width, (unsigned int)height,
                  std::vector<Color>(colors));
        return img;
    }

    std::vector<Color> colors;
    unsigned int size = width * height;
    for (unsigned char *p = imgData; p != imgData + size;
         p += channels) {  // loop through each pixel
        Color col(*p, *(p + 1), *(p + 2));
        colors.push_back(col);
    }
    Image img(width, height, std::vector<Color>(colors));
    return img;
}

}  // namespace image
