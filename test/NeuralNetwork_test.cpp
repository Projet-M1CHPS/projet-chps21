#include "NeuralNetwork.hpp"
#include <gtest/gtest.h>

TEST(NeuralNetworkTest, AF_Sigmoid) {
  ASSERT_NEAR(0.5f, nnet::sigmoid<float>(0), 0.005);
  ASSERT_NEAR(0.731f, nnet::sigmoid<float>(1), 0.005);
  ASSERT_NEAR(0.268f, nnet::sigmoid<float>(-1), 0.005);

  ASSERT_NEAR(0.5, nnet::sigmoid<double>(0), 0.005);
  ASSERT_NEAR(0.731, nnet::sigmoid<double>(1), 0.005);
  ASSERT_NEAR(0.268, nnet::sigmoid<float>(-1), 0.005);
}