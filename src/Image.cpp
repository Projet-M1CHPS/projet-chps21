#include "Image.hpp"

#include <dirent.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace image;

Color::Color(color_t r, color_t g, color_t b) {
    this->r = r;
    this->g = g;
    this->b = b;
}

void Color::print() {
    std::cout << "[" << (unsigned int)this->r << ";" << (unsigned int)this->g
              << ";" << (unsigned int)this->b << "]";
}

Image::Image() {
    this->width = 0;
    this->height = 0;
}

Image::Image(unsigned width, unsigned height, std::vector<Color> colors) {
    this->width = width;
    this->height = height;
    this->colors = colors;
}

void Image::print() {
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


Image ImageLoader::load(std::string const filename) {
    std::ifstream fp;
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
    assert(max_color >= 0 && max_color < 256);
    fp.get();
    //std::cout << "width: " << width << "; height: " << height
    //          << "; max_color: " << max_color << std::endl;
    std::vector<Color> colors;
    int r, g, b = 0;
    for (unsigned i = 0; i < height; i++) {
        for (unsigned j = 0; j < width; j++) {
            char tmp[3];
            fp >> r;
            fp >> g;
            fp >> b;
            Color col((color_t)r, (color_t)g, (color_t)b);
            // col.print();
            colors.push_back(col);
        }
    }
    Image img(width, height, colors);
    //img.print();
    fp.close();
    return img;
}



