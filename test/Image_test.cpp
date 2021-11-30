#include "Image.hpp"

#include <gtest/gtest.h>

TEST(ImageTest, DifferenceBetweenDifferentImageDimensions) {
  image::GrayscaleImage a = image::ImageSerializer::createRandomNoiseImage(124, 108);
  image::GrayscaleImage b = image::ImageSerializer::createRandomNoiseImage(96, 32);
  // Add a simple transformation
  double diff = a.getDifference(b);
  ASSERT_NO_THROW(diff);
}

TEST(ImageTest, SaveAndLoadRandomImage) {
    image::GrayscaleImage a = image::ImageSerializer::createRandomNoiseImage(128,128);
    image::ImageSerializer::save("testImage.png", a);
    image::GrayscaleImage b = image::ImageSerializer::load("testImage.png");
    ASSERT_EQ(a.getDifference(b), 0.0);
    std::remove("testImage.png");
}

TEST(ImageTest, LoadPngDirectory) { // TODO: make cleaner version
    image::GrayscaleImage a = image::ImageSerializer::createRandomNoiseImage(128,128);
    image::GrayscaleImage b = image::ImageSerializer::createRandomNoiseImage(128,128);
    image::GrayscaleImage c = image::ImageSerializer::createRandomNoiseImage(128,128);
    image::GrayscaleImage d = image::ImageSerializer::createRandomNoiseImage(128,128);
    image::GrayscaleImage e = image::ImageSerializer::createRandomNoiseImage(128,128);

    image::ImageSerializer::save("testImageA.png", a);
    image::ImageSerializer::save("testImageB.png", b);
    image::ImageSerializer::save("testImageC.png", c);
    image::ImageSerializer::save("testImageD.png", d);
    image::ImageSerializer::save("testImageE.png", e);

    std::vector<image::GrayscaleImage> img_list =  image::ImageSerializer::loadDirectory(".");
    ASSERT_EQ(img_list.size(), 5);

    std::remove("testImageA.png");
    std::remove("testImageB.png");
    std::remove("testImageC.png");
    std::remove("testImageD.png");
    std::remove("testImageE.png");
}

/* TESTS TO ADD
 *
 * - Constructors
 * -- init a grayscale image from another image, assert they arent different
 * 
 * - Getters
 * -- getPixel 1d and 2d
 * -- getData
 * -- getHistogramm
 * -- begins, ends, assert the complete iteration through the image
 * -- getDimension, size, width and height
 * 
 * - Set
 * -- change value of several pixels, assert they match the wanted values
 *
 * - Errors
 * -- add and test errors in 
 */ 