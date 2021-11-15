#include "Image.hpp"

#include <gtest/gtest.h>

TEST(ImageTest, DifferenceBetweenDifferentImageDimensions) {
    image::GrayscaleImage a = image::ImageLoader::createRandomNoiseImage(124, 108);
    image::GrayscaleImage b = image::ImageLoader::createRandomNoiseImage(96, 32);
    // Add a simple transformation
    ASSERT_NO_THROW(a.getDifference(b));
}
