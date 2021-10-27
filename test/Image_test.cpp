#include "Image.hpp"

#include <gtest/gtest.h>

TEST(ImageTest, ZeroSizedImage) {
    ASSERT_ANY_THROW(image::Image(0, 0, std::vector<image::Color>()));
}

TEST(ImageTest, NegativeSizedImage) {
    ASSERT_ANY_THROW(image::Image(-1, -1, std::vector<image::Color>()));
}