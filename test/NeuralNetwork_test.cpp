#include "NeuralNetwork.hpp"
#include <gtest/gtest.h>

#include <vector>

TEST(ActivationFunctionTest, AF_Sigmoid) {
  ASSERT_NEAR(0.5f, af::sigmoid<float>(0), 0.005);
  ASSERT_NEAR(0.731f, af::sigmoid<float>(1), 0.005);
  ASSERT_NEAR(0.268f, af::sigmoid<float>(-1), 0.005);

  ASSERT_NEAR(0.5, af::sigmoid<double>(0), 0.005);
  ASSERT_NEAR(0.731, af::sigmoid<double>(1), 0.005);
  ASSERT_NEAR(0.268, af::sigmoid<float>(-1), 0.005);
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

TEST(NeuralNetworkTest, CanSetLayerSize) {
  nnet::NeuralNetwork<float> nn;
  std::vector<size_t> layer_size = {1, 2};
  nn.setLayersSize(layer_size);

  ASSERT_EQ(2, nn.getOutputSize());
  ASSERT_EQ(1, nn.getInputSize());
}

TEST(NeuralNetworkTest, ThrowOnInvalidLayerSize) {
  nnet::NeuralNetwork<float> nn;
  std::vector<size_t> layer_size = {1};

  ASSERT_ANY_THROW(nn.setLayersSize(layer_size));
}

TEST(NeuralNetworkTest, CanCopyNeuralNetwork) {
  nnet::NeuralNetwork<float> nn;
  std::vector<size_t> layer_size = {1, 2};
  nn.setLayersSize(layer_size);

  nn.randomizeSynapses();

  // copy the network
  nnet::NeuralNetwork<float> nn2(nn);

  ASSERT_EQ(2, nn2.getOutputSize());
  ASSERT_EQ(1, nn2.getInputSize());

  auto weights = nn2.getWeights();
  auto weights2 = nn.getWeights();

  // Check if all the weights are the same
  ASSERT_EQ(weights2.size(), weights.size());

  for (size_t i = 0; i < weights.size(); i++) {
    ASSERT_EQ(weights[i].getRows(), weights2[i].getRows());
    ASSERT_EQ(weights[i].getCols(), weights2[i].getCols());

    float *data = weights[i].getData();
    float *data2 = weights2[i].getData();

    for (size_t j = 0; j < weights[i].getCols() * weights[i].getRows(); j++) {
      ASSERT_EQ(data[j], data2[j]);
    }
  }
}

TEST(NeuralNetworkPrecisionTest, CanConvertStrToPrecision) {
  ASSERT_EQ(nnet::FloatingPrecision::float32, nnet::strToFPrecision("float32"));
  ASSERT_EQ(nnet::FloatingPrecision::float64, nnet::strToFPrecision("float64"));
}

TEST(NeuralNetworkTest, ThrowOnInvalidInput) {
  nnet::NeuralNetwork<float> nn;
  nn.setLayersSize(std::vector<size_t>{2, 2, 1});
  std::vector<float> input = {1, 2, 3, 4};

  ASSERT_ANY_THROW(nn.forward(input.begin(), input.end()));
}

TEST(NeuralNetworkTest, SimpleNeuralTest) {

  nnet::NeuralNetwork<float> nn;
  nn.setLayersSize(std::vector<size_t>{2, 2, 1});
  nn.setActivationFunction(af::ActivationFunctionType::square);

  auto &w = nn.getWeights();
  auto &b = nn.getBiases();

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
  nn.setActivationFunction(af::ActivationFunctionType::square);

  auto &w = nn.getWeights();
  auto &b = nn.getBiases();

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