#include "Transform.hpp"

#include <gtest/gtest.h>

#include "Image.hpp"

TEST(TransformTest, HistogramInversionLossless) {
    image::GrayscaleImage original = image::ImageLoader::createRandomNoiseImage();
    // Add a simple transformation
    image::transform::TransformEngine te;
    te.addTransformation(
        std::make_shared<image::transform::HistogramInversion>());
    te.addTransformation(
        std::make_shared<image::transform::HistogramInversion>());
    image::GrayscaleImage edited = te.transform(original);
    ASSERT_FLOAT_EQ(original.difference(edited), .0);
}

TEST(TransformTest, DoubleGreyScaleLossless) {
    image::GrayscaleImage original = image::ImageLoader::createRandomNoiseImage();
    // Add a simple transformation
    image::transform::TransformEngine te;
    te.addTransformation(std::make_shared<image::transform::GreyScale>());
    image::GrayscaleImage greyscaled_once = te.transform(original);
    image::GrayscaleImage greyscaled_twice = te.transform(greyscaled_once);
    ASSERT_FLOAT_EQ(greyscaled_once.difference(greyscaled_twice), .0);
}