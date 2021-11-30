#include "Image.hpp"

#include <gtest/gtest.h>

TEST(ImageTest, DifferenceBetweenDifferentImageDimensions) {
  image::GrayscaleImage a = image::ImageSerializer::createRandomNoiseImage(124, 108);
  image::GrayscaleImage b = image::ImageSerializer::createRandomNoiseImage(96, 32);
  // Add a simple transformation
  double diff = a.getDifference(b);
  ASSERT_NO_THROW(diff);
}
