#include "Transform.hpp"

#include <gtest/gtest.h>

#include "Image.hpp"

TEST(TransformTest, HistogramInversionLossless) {
    image::GrayscaleImage original = image::ImageSerializer::createRandomNoiseImage();
    // Add a simple transformation
    image::transform::TransformEngine te;
    te.addTransformation(
        std::make_shared<image::transform::Inversion>());
    te.addTransformation(
        std::make_shared<image::transform::Inversion>());
    image::GrayscaleImage edited = te.transform(original);
    ASSERT_FLOAT_EQ(original.getDifference(edited), .0);
}

TEST(TransformTest, DoubleBinaryScaleLossless) {
    image::GrayscaleImage original = image::ImageSerializer::createRandomNoiseImage();
    // Add a simple transformation
    image::transform::TransformEngine te;
    te.addTransformation(std::make_shared<image::transform::BinaryScale>());
    image::GrayscaleImage binarised_once = te.transform(original);
    image::GrayscaleImage binarised_twice = te.transform(binarised_once);
    ASSERT_FLOAT_EQ(binarised_once.getDifference(binarised_twice), .0);
}
