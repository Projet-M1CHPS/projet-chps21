#include "Image.hpp"

#include <gtest/gtest.h>

TEST(ImageTest, DifferenceBetweenDifferentImageDimensions) {
    image::GrayscaleImage a = image::ImageSerializer::createRandomNoiseImage(124, 108);
    image::GrayscaleImage b = image::ImageSerializer::createRandomNoiseImage(96, 32);
    // Add a simple transformation
    ASSERT_NO_THROW(a.getDifference(b));
}

TEST(ImageTest, SaveAndLoadRandomImage) {
    std::filesystem::create_directory("SaveAndLoadRandomImage_TestDir"); //create a dir to store the test images

    image::GrayscaleImage a = image::ImageSerializer::createRandomNoiseImage(128,128);
    image::ImageSerializer::save("SaveAndLoadRandomImage_TestDir/testImage.png", a);
    image::GrayscaleImage b = image::ImageSerializer::load("SaveAndLoadRandomImage_TestDir/testImage.png");
    ASSERT_EQ(a.getDifference(b), 0.0);

    std::filesystem::remove_all("SaveAndLoadRandomImage_TestDir"); // delete the test dir
}

TEST(ImageTest, LoadPngDirectory) {
    std::filesystem::create_directory("LoadPngDirectory_TestDir"); //create a dir to store the test images

    image::GrayscaleImage a = image::ImageSerializer::createRandomNoiseImage(128,128);
    image::GrayscaleImage b = image::ImageSerializer::createRandomNoiseImage(128,128);
    image::GrayscaleImage c = image::ImageSerializer::createRandomNoiseImage(128,128);
    image::GrayscaleImage d = image::ImageSerializer::createRandomNoiseImage(128,128);
    image::GrayscaleImage e = image::ImageSerializer::createRandomNoiseImage(128,128);

    image::ImageSerializer::save("LoadPngDirectory_TestDir/testImageA.png", a);
    image::ImageSerializer::save("LoadPngDirectory_TestDir/testImageB.png", b);
    image::ImageSerializer::save("LoadPngDirectory_TestDir/testImageC.png", c);
    image::ImageSerializer::save("LoadPngDirectory_TestDir/testImageD.png", d);
    image::ImageSerializer::save("LoadPngDirectory_TestDir/testImageE.png", e);

    std::vector<image::GrayscaleImage> img_list =  image::ImageSerializer::loadDirectory("LoadPngDirectory_TestDir");
    ASSERT_EQ(img_list.size(), 5);

    std::filesystem::remove_all("LoadPngDirectory_TestDir"); // delete the test dir
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