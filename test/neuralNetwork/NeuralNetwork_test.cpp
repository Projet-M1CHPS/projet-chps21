#include "NeuralNetwork.hpp"
#include "Optimizer.hpp"
#include "TrainingMethod.hpp"
#include <gtest/gtest.h>

#include <vector>


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
  std::vector<size_t> layer_size = {2, 1};
  nn.setLayersSize(layer_size);

  ASSERT_EQ(2, nn.getInputSize());
  ASSERT_EQ(1, nn.getOutputSize());
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

  // copy the network with constructor
  nnet::NeuralNetwork<float> nn2(nn);

  ASSERT_EQ(2, nn2.getOutputSize());
  ASSERT_EQ(1, nn2.getInputSize());

  auto weights = nn.getWeights();
  auto biases = nn.getBiases();
  auto weights2 = nn2.getWeights();
  auto biases2 = nn2.getBiases();

  // Check if all the weights are the same
  ASSERT_EQ(weights.size(), weights2.size());
  ASSERT_EQ(biases.size(), biases2.size());

  for (size_t i = 0; i < weights.size(); i++) {
    ASSERT_EQ(weights[i].getRows(), weights2[i].getRows());
    ASSERT_EQ(weights[i].getCols(), weights2[i].getCols());

    ASSERT_EQ(biases[i].getRows(), biases2[i].getRows());
    ASSERT_EQ(biases[i].getCols(), biases2[i].getCols());

    float *data_weights = weights[i].getData();
    float *data_weights2 = weights2[i].getData();
    float *data_biases = biases[i].getData();
    float *data_biases2 = biases2[i].getData();

    for (size_t j = 0; j < weights[i].getCols() * weights[i].getRows(); j++) {
      ASSERT_EQ(data_weights[j], data_weights2[j]);
    }
    for (size_t j = 0; j < biases[i].getCols() * biases[i].getRows(); j++) {
      ASSERT_EQ(data_biases[j], data_biases2[j]);
    }
  }

  // copy the network with operator
  nnet::NeuralNetwork<float> nn3 = nn;

  ASSERT_EQ(2, nn3.getOutputSize());
  ASSERT_EQ(1, nn3.getInputSize());

  // weights = nn.getWeights();
  // biases = nn.getBiases();
  weights2 = nn3.getWeights();
  biases2 = nn3.getBiases();

  // Check if all the weights are the same
  ASSERT_EQ(weights2.size(), weights.size());

  for (size_t i = 0; i < weights.size(); i++) {
    ASSERT_EQ(weights[i].getRows(), weights2[i].getRows());
    ASSERT_EQ(weights[i].getCols(), weights2[i].getCols());

    ASSERT_EQ(biases[i].getRows(), biases2[i].getRows());
    ASSERT_EQ(biases[i].getCols(), biases2[i].getCols());

    float *data_weights = weights[i].getData();
    float *data_weights2 = weights2[i].getData();
    float *data_biases = biases[i].getData();
    float *data_biases2 = biases2[i].getData();

    for (size_t j = 0; j < weights[i].getCols() * weights[i].getRows(); j++) {
      ASSERT_EQ(data_weights[j], data_weights2[j]);
    }
    for (size_t j = 0; j < biases[i].getCols() * biases[i].getRows(); j++) {
      ASSERT_EQ(data_biases[j], data_biases2[j]);
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
  math::Matrix<float> input = {1, 2, 3, 4};

  ASSERT_ANY_THROW(nn.predict(input));
}

TEST(NeuralNetworkTest, ThrowOnInvalidInputOrTarget) {
  nnet::NeuralNetwork<float> nn1;
  nn1.setLayersSize(std::vector<size_t>{2, 2, 1});

  nnet::StandardTrainingMethod<float> stdTrain1(0.1);
  nnet::MLPStochOptimizer<float> opti1(&nn1, &stdTrain1);

  math::Matrix<float> input1 = {1, 2, 3, 4};
  math::Matrix<float> target1 = {1};

  ASSERT_ANY_THROW(opti1.train(input1, target1));

  nnet::NeuralNetwork<float> nn2;
  nn2.setLayersSize(std::vector<size_t>{2, 2, 1});
  math::Matrix<float> input2 = {1, 2};
  math::Matrix<float> target2 = {1, 2, 3};

  nnet::StandardTrainingMethod<float> stdTrain2(0.1);
  nnet::MLPStochOptimizer<float> opti2(&nn2, &stdTrain2);

  ASSERT_ANY_THROW(opti2.train(input2, target2));
}


TEST(NeuralNetworkTest, SimpleNeuralTest) {
  nnet::NeuralNetwork<float> nn;
  nn.setLayersSize(std::vector<size_t>{2, 2, 1});
  nn.setActivationFunction(af::ActivationFunctionType::square);

  auto &w = nn.getWeights();
  auto &b = nn.getBiases();

  for (auto &i : w) {
    for (auto &e : i) { e = 1.f; }
  }

  for (auto &i : b) {
    for (auto &e : i) { e = 1.f; }
  }

  math::Matrix<float> input{1, 1};
  auto output = nn.predict(input);

  ASSERT_NEAR(361.f, output(0, 0), 0.005);
}

TEST(NeuralNetworkTest, ComplexNeuralTest) {
  nnet::NeuralNetwork<double> nn;
  nn.setLayersSize(std::vector<size_t>{2, 4, 2, 3, 2});
  nn.setActivationFunction(af::ActivationFunctionType::relu);

  auto &w = nn.getWeights();
  auto &b = nn.getBiases();

  for (auto &i : w) {
    for (auto &e : i) { e = 1.f; }
  }

  for (auto &i : b) {
    for (auto &e : i) { e = 1.f; }
  }

  math::Matrix<double> input{1, 1};
  auto output = nn.predict(input);

  ASSERT_NEAR(82.f, output(0, 0), 0.005);
  ASSERT_NEAR(82.f, output(1, 0), 0.005);

  ASSERT_EQ(output.getRows(), 2);
  ASSERT_EQ(output.getCols(), 1);
}

TEST(NeuralNetworkTest, OtherComplexNeuralTest) {
  nnet::NeuralNetwork<float> nn;
  nn.setLayersSize(std::vector<size_t>{2, 2, 2});
  nn.setActivationFunction(af::ActivationFunctionType::sigmoid);

  nnet::StandardTrainingMethod<float> stdTrain(0.5);
  nnet::MLPStochOptimizer<float> opti(&nn, &stdTrain);

  math::Matrix<float> &w1 = nn.getWeights()[0];
  math::Matrix<float> &b1 = nn.getBiases()[0];
  math::Matrix<float> &w2 = nn.getWeights()[1];
  math::Matrix<float> &b2 = nn.getBiases()[1];

  w1(0, 0) = 0.15;   // w1
  w1(0, 1) = 0.20;   // w3
  w1(1, 0) = 0.25;   // w2
  w1(1, 1) = 0.30;   // w4
  b1(0, 0) = 0.35;   // b1
  b1(1, 0) = 0.35;   // b2

  w2(0, 0) = 0.40;   // w1
  w2(0, 1) = 0.45;   // w3
  w2(1, 0) = 0.50;   // w2
  w2(1, 1) = 0.55;   // w4
  b2(0, 0) = 0.60;   // b1
  b2(1, 0) = 0.60;   // b2

  math::Matrix<float> input{0.05, 0.10};
  math::Matrix<float> output{0.01, 0.99};

  auto prediction = nn.predict(input);
  ASSERT_NEAR(0.751365f, prediction(0, 0), 0.005);
  ASSERT_NEAR(0.772928f, prediction(1, 0), 0.005);
  ASSERT_EQ(prediction.getRows(), 2);
  ASSERT_EQ(prediction.getCols(), 1);

  opti.train(input, output);

  math::Matrix<float> &w1_ = nn.getWeights()[0];
  math::Matrix<float> &b1_ = nn.getBiases()[0];
  math::Matrix<float> &w2_ = nn.getWeights()[1];
  math::Matrix<float> &b2_ = nn.getBiases()[1];

  //
  ASSERT_NEAR(0.149781f, w1_(0, 0), 0.005);
  ASSERT_NEAR(0.199561f, w1_(0, 1), 0.005);
  ASSERT_NEAR(0.249751f, w1_(1, 0), 0.005);
  ASSERT_NEAR(0.299502f, w1_(1, 1), 0.005);

  //
  ASSERT_NEAR(0.358916f, w2_(0, 0), 0.005);
  ASSERT_NEAR(0.408666f, w2_(0, 1), 0.005);
  ASSERT_NEAR(0.511301f, w2_(1, 0), 0.005);
  ASSERT_NEAR(0.561378f, w2_(1, 1), 0.005);
}