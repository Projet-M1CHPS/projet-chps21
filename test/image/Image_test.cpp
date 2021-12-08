#include "Image.hpp"

#include <gtest/gtest.h>

TEST(ImageTest, DifferenceBetweenDifferentImageDimensions) {
    image::GrayscaleImage a = image::ImageSerializer::createRandomNoiseImage(124, 108);
    image::GrayscaleImage b = image::ImageSerializer::createRandomNoiseImage(96, 32);
    // Add a simple transformation
    ASSERT_NO_THROW(a.getDifference(b));
}
