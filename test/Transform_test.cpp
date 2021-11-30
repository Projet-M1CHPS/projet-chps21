#include "Transform.hpp"

#include <gtest/gtest.h>

#include "Image.hpp"

TEST(TransformTest, HistogramInversionLossless) {
  image::GrayscaleImage original = image::ImageSerializer::createRandomNoiseImage();
  // Add a simple transformation
  image::transform::TransformEngine te;
  te.addTransformation(std::make_shared<image::transform::Inversion>());
  te.addTransformation(std::make_shared<image::transform::Inversion>());
  image::GrayscaleImage edited = te.transform(original);
  double diff = original.getDifference(edited);
  ASSERT_FLOAT_EQ(diff, .0);
}

TEST(TransformTest, DoubleBinaryScaleLossless) {
  image::GrayscaleImage original = image::ImageSerializer::createRandomNoiseImage();
  // Add a simple transformation
  image::transform::TransformEngine te;
  te.addTransformation(std::make_shared<image::transform::BinaryScale>());
  image::GrayscaleImage binarised_once = te.transform(original);
  image::GrayscaleImage binarised_twice = te.transform(binarised_once);
  double diff = binarised_once.getDifference(binarised_twice);
  ASSERT_FLOAT_EQ(diff, .0);
}

TEST(TransformTest, ResizeInSquare) {
  image::GrayscaleImage original = image::ImageSerializer::createRandomNoiseImage(50, 50);
  // Add a simple transformation
  image::transform::Resize resize_tranform = image::transform::Resize(10, 10);
  ASSERT_NO_THROW(resize_tranform.transform(original));
  ASSERT_TRUE(original.getWidth() == 10 && original.getHeight() == 10);
}


TEST(TransformTest, ResizeInRectangle) {
  image::GrayscaleImage original = image::ImageSerializer::createRandomNoiseImage(50, 50);
  // Add a simple transformation
  image::transform::Resize resize_tranform = image::transform::Resize(10, 25);
  ASSERT_NO_THROW(resize_tranform.transform(original));
  ASSERT_TRUE(original.getWidth() == 10 && original.getHeight() == 25);
}

TEST(TransformTest, ResizeToZero) {
  image::GrayscaleImage original = image::ImageSerializer::createRandomNoiseImage(50, 50);
  // Add a simple transformation
  image::transform::Resize resize_tranform = image::transform::Resize(0, 25);
  ASSERT_THROW(resize_tranform.transform(original), std::invalid_argument);
  resize_tranform = image::transform::Resize(25, 0);
  ASSERT_THROW(resize_tranform.transform(original), std::invalid_argument);
  resize_tranform = image::transform::Resize(0, 0);
  ASSERT_THROW(resize_tranform.transform(original), std::invalid_argument);
}
