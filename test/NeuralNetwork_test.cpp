#include "NeuralNetwork.hpp"
#include <gtest/gtest.h>

#include <vector>

TEST(ActivationFunctionTest, AF_Sigmoid) {
  ASSERT_NEAR(0.5f, nnet::sigmoid<float>(0), 0.005);
  ASSERT_NEAR(0.731f, nnet::sigmoid<float>(1), 0.005);
  ASSERT_NEAR(0.268f, nnet::sigmoid<float>(-1), 0.005);

  ASSERT_NEAR(0.5, nnet::sigmoid<double>(0), 0.005);
  ASSERT_NEAR(0.731, nnet::sigmoid<double>(1), 0.005);
  ASSERT_NEAR(0.268, nnet::sigmoid<float>(-1), 0.005);
}

TEST(NeuralNetworkTest, CanCreateNeuralNetwork) {
  nnet::NeuralNetwork<float> nn;

  ASSERT_EQ(0, nn.getOutputSize());
  ASSERT_EQ(0, nn.getInputSize());
  ASSERT_EQ(nnet::FloatingPrecision::float32, nn.getPrecision());

  nnet::NeuralNetwork<double> nn2;

  ASSERT_EQ(0, nn2.getOutputSize());
  ASSERT_EQ(0, nn2.getInputSize());
  ASSERT_EQ(nnet::FloatingPrecision::float64, nn2.getPrecision());
}

TEST(NeuralNetworkTest, ThrowOnInvalidLayerSize) {
  nnet::NeuralNetwork<float> nn;
  std::vector<size_t> layer_size = {1};

  ASSERT_ANY_THROW(nn.setLayersSize(layer_size));
}

TEST(NeuralNetworkTest, CanCopyNeuralNetwork) {}

TEST(NeuralNetworkTest, CanSetActivationFunction) {}

TEST(NeuralNetworkTest, ThrowOnInvalidActivationFunction) {}

TEST(NeuralNetworkTest, SimpleNeuralTest) {

  nnet::NeuralNetwork<float> nn;
  nn.setLayersSize(std::vector<size_t>{2, 2, 1});
  nn.setActivationFunction(nnet::ActivationFunctionType::square);

  auto &w = nn.getWeights();
  auto &b = nn.getBiaises();

  for (auto &i : w) {
    for (auto &e : i) {
      e = 1.f;
    }
  }

  for (auto &i : b) {
    for (auto &e : i) {
      e = 1.f;
    }
  }

  std::vector<float> input{1, 1};
  auto output = nn.forward(input.begin(), input.end());

  ASSERT_NEAR(7.f, output(0, 0), 0.005);
}

TEST(NeuralNetworkTest, ComplexNeuralTest) {

  nnet::NeuralNetwork<double> nn;
  nn.setLayersSize(std::vector<size_t>{2, 4, 2, 3, 2});
  nn.setActivationFunction(nnet::ActivationFunctionType::square);

  auto &w = nn.getWeights();
  auto &b = nn.getBiaises();

  for (auto &i : w) {
    for (auto &e : i) {
      e = 1.f;
    }
  }

  for (auto &i : b) {
    for (auto &e : i) {
      e = 1.f;
    }
  }

  std::vector<double> input{1, 1};
  auto output = nn.forward(input.begin(), input.end());

  ASSERT_NEAR(82.f, output(0, 0), 0.005);
  ASSERT_NEAR(82.f, output(0, 1), 0.005);
}