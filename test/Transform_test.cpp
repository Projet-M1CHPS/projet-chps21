#include "Transform.hpp"

#include <gtest/gtest.h>

#include "Image.hpp"

TEST(TransformTest, HistogramInversionLossless) {
    image::Image original = image::ImageLoader::createRandomImage();
    // Add a simple transformation
    image::transform::TransformEngine te;
    te.addTransformation(
        std::make_shared<image::transform::HistogramInversion>());
    te.addTransformation(
        std::make_shared<image::transform::HistogramInversion>());
    image::Image edited = te.transform(original);
    ASSERT_FLOAT_EQ(original.difference(edited), .0);
}

TEST(TransformTest, DoubleGreyScaleLossless) {
    image::Image original = image::ImageLoader::createRandomImage();
    // Add a simple transformation
    image::transform::TransformEngine te;
    te.addTransformation(std::make_shared<image::transform::GreyScale>());
    image::Image greyscaled_once = te.transform(original);
    image::Image greyscaled_twice = te.transform(greyscaled_once);
    ASSERT_FLOAT_EQ(greyscaled_once.difference(greyscaled_twice), .0);
}