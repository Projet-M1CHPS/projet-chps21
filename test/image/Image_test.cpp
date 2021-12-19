#include "Image.hpp"

#include <gtest/gtest.h>


TEST(ImageTest, CreateImageFromAnotherImage) {
    image::GrayscaleImage a = image::ImageSerializer::createRandomNoiseImage(128, 128);
    image::GrayscaleImage b{a};
    ASSERT_EQ(a.getDifference(b), 0.0);
}

TEST(ImageTest, DifferenceBetweenDifferentImageDimensions) {
  image::GrayscaleImage a = image::ImageSerializer::createRandomNoiseImage(124, 108);
  image::GrayscaleImage b = image::ImageSerializer::createRandomNoiseImage(96, 32);
  // Add a simple transformation
  double diff = a.getDifference(b);
  ASSERT_NO_THROW(diff);
}

TEST(ImageTest, SaveAndLoadRandomImage) {
    std::filesystem::create_directory("SaveAndLoadRandomImage_TestDir"); //create a dir to store the test images

    image::GrayscaleImage a = image::ImageSerializer::createRandomNoiseImage(128,128);
    image::ImageSerializer::save("SaveAndLoadRandomImage_TestDir/testImage.png", a);
    image::GrayscaleImage b = image::ImageSerializer::load("SaveAndLoadRandomImage_TestDir/testImage.png");
    ASSERT_EQ(a.getDifference(b), 0.0); // the images are similar -> save and load are working
    ASSERT_ANY_THROW(image::ImageSerializer::load("thisImageDoesntExist"));
    std::filesystem::remove_all("SaveAndLoadRandomImage_TestDir"); // delete the test dir
}

TEST(ImageTest, LoadPngDirectory) {
    
    ASSERT_ANY_THROW(image::ImageSerializer::loadDirectory("Image_test")); // error, the file isnt a directory
    ASSERT_ANY_THROW(image::ImageSerializer::loadDirectory("LoadPngDirectory_TestDir")); // error, the directory doesnt exist
    std::filesystem::create_directory("LoadPngDirectory_TestDir"); //create a dir to store the test images

    ASSERT_ANY_THROW(image::ImageSerializer::loadDirectory("LoadPngDirectory_TestDir")); // warning, the directory is empty

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

TEST(ImageTest, ImageGetters) {
    image::GrayscaleImage a = image::ImageSerializer::createRandomNoiseImage(128,64);
    // test most getters
    ASSERT_EQ(a.getWidth(), 128);
    ASSERT_EQ(a.getHeight(), 64);
    ASSERT_EQ(a.getSize(), 128 * 64);
    ASSERT_EQ(a.getDimension().first,128);
    ASSERT_EQ(a.getDimension().second,64);
    image::grayscale_t* pixel_data_ptr = a.getData();
    for (unsigned i = 0; i < a.getSize(); i++) {
        ASSERT_EQ(pixel_data_ptr[i], a.getPixel(i));
    }

    // special histogram tests
    std::unique_ptr<image::grayscale_t[]> pixel_data_ptr2 = std::make_unique<image::grayscale_t[]>(6);
    pixel_data_ptr2[0] = 0;
    pixel_data_ptr2[1] = 255;
    pixel_data_ptr2[2] = 0;
    pixel_data_ptr2[3] = 1;
    pixel_data_ptr2[4] = 0;
    pixel_data_ptr2[5] = 255;
    image::GrayscaleImage b{6,1,std::move(pixel_data_ptr2)};
    std::vector<size_t> bHist = b.getHistogram();
    ASSERT_EQ(bHist[0], 3);
    ASSERT_EQ(bHist[1], 1);
    ASSERT_EQ(bHist[255], 2);
}

TEST(ImageTest, ImageSetters) {
    std::unique_ptr<image::grayscale_t[]> pixel_data = std::make_unique<image::grayscale_t[]>(4);
    pixel_data[0] = 0;
    pixel_data[1] = 0;
    pixel_data[2] = 0;
    pixel_data[3] = 0;
    image::GrayscaleImage a{2,2,std::move(pixel_data)};

    image::grayscale_t* pixel_data_ptr = a.getData();
    // changes and assert the change of some pixels
    pixel_data_ptr[0] = 255;
    pixel_data_ptr[2] = 255;
    ASSERT_EQ(a.getPixel(0), 255);
    ASSERT_EQ(a.getPixel(0,1), 255);

    // resize the image (values arent kept)
    a.setSize(4,4);
    ASSERT_EQ(a.getWidth(), 4);
    ASSERT_EQ(a.getHeight(), 4);
    ASSERT_EQ(a.getSize(), 4*4);
}