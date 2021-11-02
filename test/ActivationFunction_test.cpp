#include "ActivationFunction.hpp"
#include <gtest/gtest.h>

TEST(ActivationFunctionTest, AF_Identity) {
  ASSERT_NEAR(0.6f, af::identity<float>(0.6), 0.005);
  ASSERT_NEAR(1.1f, af::identity<float>(1.1), 0.005);
  ASSERT_NEAR(1.f, af::didentity<float>(0.8), 0.005);

  ASSERT_NEAR(0.6, af::identity<double>(0.6), 0.005);
  ASSERT_NEAR(1.1, af::identity<double>(1.1), 0.005);
  ASSERT_NEAR(1.0, af::didentity<double>(0.8), 0.005);
}

TEST(ActivationFunctionTest, AF_Sigmoid) {
  ASSERT_NEAR(0.645f, af::sigmoid<float>(0.6), 0.005);
  ASSERT_NEAR(0.750f, af::sigmoid<float>(1.1), 0.005);
  ASSERT_NEAR(0.213f, af::dsigmoid<float>(0.8), 0.005);

  ASSERT_NEAR(0.645, af::sigmoid<double>(0.6), 0.005);
  ASSERT_NEAR(0.750, af::sigmoid<double>(1.1), 0.005);
  ASSERT_NEAR(0.213, af::dsigmoid<double>(0.8), 0.005);
}

TEST(ActivationFunctionTest, AF_Relu) {
  ASSERT_NEAR(0.6f, af::relu<float>(0.6), 0.005);
  ASSERT_NEAR(0.f, af::relu<float>(-1.1), 0.005);
  ASSERT_NEAR(1.f, af::drelu<float>(0.8), 0.005);
  ASSERT_NEAR(0.f, af::drelu<float>(-0.2), 0.005);

  ASSERT_NEAR(0.6, af::relu<double>(0.6), 0.005);
  ASSERT_NEAR(0.0, af::relu<double>(-1.1), 0.005);
  ASSERT_NEAR(1.0, af::drelu<double>(0.8), 0.005);
  ASSERT_NEAR(0.0, af::drelu<double>(-0.2), 0.005);
}

TEST(ActivationFunctionTest, AF_LeakyRelu) {
  ASSERT_NEAR(0.6f, af::leakyRelu<float>(0.6), 0.005);
  ASSERT_NEAR(-0.011f, af::leakyRelu<float>(-1.1), 0.005);
  ASSERT_NEAR(1.f, af::dleakyRelu<float>(0.8), 0.005);
  ASSERT_NEAR(0.01f, af::dleakyRelu<float>(-0.2), 0.005);

  ASSERT_NEAR(0.6, af::leakyRelu<double>(0.6), 0.005);
  ASSERT_NEAR(-0.011, af::leakyRelu<double>(-1.1), 0.005);
  ASSERT_NEAR(1.0, af::dleakyRelu<double>(0.8), 0.005);
  ASSERT_NEAR(0.01, af::dleakyRelu<double>(-0.2), 0.005);
}

TEST(ActivationFunctionTest, AF_Square) {
  ASSERT_NEAR(0.36f, af::square<float>(0.6), 0.005);
  ASSERT_NEAR(1.21f, af::square<float>(1.1), 0.005);
  ASSERT_NEAR(1.6f, af::dsquare<float>(0.8), 0.005);

  ASSERT_NEAR(0.36, af::square<double>(0.6), 0.005);
  ASSERT_NEAR(1.21, af::square<double>(1.1), 0.005);
  ASSERT_NEAR(1.6, af::dsquare<double>(0.8), 0.005);
}