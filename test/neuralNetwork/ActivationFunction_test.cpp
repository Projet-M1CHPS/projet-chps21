#include "ActivationFunction.hpp"
#include <gtest/gtest.h>

TEST(ActivationFunctionTest, AF_Identity) {
  ASSERT_NEAR(0.6f, af::identity(0.6), 0.005);
  ASSERT_NEAR(1.1f, af::identity(1.1), 0.005);
  ASSERT_NEAR(1.f, af::didentity(0.8), 0.005);

  ASSERT_NEAR(0.6, af::identity(0.6), 0.005);
  ASSERT_NEAR(1.1, af::identity(1.1), 0.005);
  ASSERT_NEAR(1.0, af::didentity(0.8), 0.005);
}

TEST(ActivationFunctionTest, AF_Sigmoid) {
  ASSERT_NEAR(0.645f, af::sigmoid(0.6), 0.005);
  ASSERT_NEAR(0.750f, af::sigmoid(1.1), 0.005);
  ASSERT_NEAR(0.213f, af::dsigmoid(0.8), 0.005);

  ASSERT_NEAR(0.645, af::sigmoid(0.6), 0.005);
  ASSERT_NEAR(0.750, af::sigmoid(1.1), 0.005);
  ASSERT_NEAR(0.213, af::dsigmoid(0.8), 0.005);
}

TEST(ActivationFunctionTest, AF_Relu) {
  ASSERT_NEAR(0.6f, af::relu(0.6), 0.005);
  ASSERT_NEAR(0.f, af::relu(-1.1), 0.005);
  ASSERT_NEAR(1.f, af::drelu(0.8), 0.005);
  ASSERT_NEAR(0.f, af::drelu(-0.2), 0.005);

  ASSERT_NEAR(0.6, af::relu(0.6), 0.005);
  ASSERT_NEAR(0.0, af::relu(-1.1), 0.005);
  ASSERT_NEAR(1.0, af::drelu(0.8), 0.005);
  ASSERT_NEAR(0.0, af::drelu(-0.2), 0.005);
}

TEST(ActivationFunctionTest, AF_LeakyRelu) {
  ASSERT_NEAR(0.6f, af::leakyRelu(0.6), 0.005);
  ASSERT_NEAR(-0.011f, af::leakyRelu(-1.1), 0.005);
  ASSERT_NEAR(1.f, af::dleakyRelu(0.8), 0.005);
  ASSERT_NEAR(0.01f, af::dleakyRelu(-0.2), 0.005);

  ASSERT_NEAR(0.6, af::leakyRelu(0.6), 0.005);
  ASSERT_NEAR(-0.011, af::leakyRelu(-1.1), 0.005);
  ASSERT_NEAR(1.0, af::dleakyRelu(0.8), 0.005);
  ASSERT_NEAR(0.01, af::dleakyRelu(-0.2), 0.005);
}

TEST(ActivationFunctionTest, AF_Square) {
  ASSERT_NEAR(0.36f, af::square(0.6), 0.005);
  ASSERT_NEAR(1.21f, af::square(1.1), 0.005);
  ASSERT_NEAR(1.6f, af::dsquare(0.8), 0.005);

  ASSERT_NEAR(0.36, af::square(0.6), 0.005);
  ASSERT_NEAR(1.21, af::square(1.1), 0.005);
  ASSERT_NEAR(1.6, af::dsquare(0.8), 0.005);
}