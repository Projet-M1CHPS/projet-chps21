
#include "NeuralNetwork.hpp"
#include "ProjectVersion.hpp"
#include "tscl.hpp"

#include "CNN.hpp"
#include "CNNLayer.hpp"
#include "CNNModel.hpp"
#include "CNNOptimizer.hpp"
#include "CNNStorageBP.hpp"
#include <curses.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>


// Really simple XOR example
// This example uses a Stochastic Gradient Descent algorithm to train a neural network to solve a
// XOR problem. This example serves as a crash test for the neural network, to be used when
// everything else fails and for the most desperate times.
void runXor(const size_t bach_size, const float learning_rate, const float error_limit) {
  nnet::MLPModel model;
  auto &nn1 = model.getPerceptron();
  nnet::MLPTopology topology = {2, 3, 3, 1};
  nn1.setTopology(topology);
  nn1.setActivationFunction(af::ActivationFunctionType::sigmoid);
  nn1.randomizeWeight();

  auto &w1 = nn1.getWeights();

  auto optimizer = nnet::MLPStochOptimizer::make<nnet::SGDOptimization>(model, learning_rate);

  std::vector<math::FloatMatrix> input_buffer(4);
  std::vector<math::FloatMatrix> target_buffer(4);
  for (size_t i = 0; i < 4; i++) {
    input_buffer[i] = math::FloatMatrix(2, 1);
    target_buffer[i] = math::FloatMatrix(1, 1);
  }

  // Xor truth table
  // Yes, this is ugly, but who cares ?
  input_buffer[0](0, 0) = 0.f;
  input_buffer[0](1, 0) = 0.f;
  input_buffer[1](0, 0) = 1.f;
  input_buffer[1](1, 0) = 0.f;
  input_buffer[2](0, 0) = 0.f;
  input_buffer[2](1, 0) = 1.f;
  input_buffer[3](0, 0) = 1.f;
  input_buffer[3](1, 0) = 1.f;

  target_buffer[0](0, 0) = 0.f;
  target_buffer[1](0, 0) = 1.f;
  target_buffer[2](0, 0) = 1.f;
  target_buffer[3](0, 0) = 0.f;

  std::vector<math::clFMatrix> input(4);
  std::vector<math::clFMatrix> target(4);
  for (size_t i = 0; i < 4; i++) {
    input[i] = input_buffer[i];
    target[i] = target_buffer[i];
  }

  float error = 1.0;
  size_t count = 0;
  std::cout << std::setprecision(8);
  while (error > error_limit && count < 600) {
    for (int i = 0; i < bach_size; i++) {
      for (int j = 0; j < 4; j++) { optimizer->optimize(input[j], target[j]); }
    }

    error = 0.0;
    for (int i = 0; i < input.size(); i++) {
      auto cl_matrix = nn1.predict(input[i]);
      auto fmatrix = cl_matrix.toFloatMatrix();
      error += std::fabs(fmatrix(0, 0) - target_buffer[i](0, 0));
    }
    error /= (float) input.size();
    std::cout << error << std::endl;
    count++;
  }

  std::cout << nn1 << std::endl;
  std::cout << "Result"
            << "---> " << count << " iterations" << std::endl;
  for (int i = 0; i < input.size(); i++) {
    auto cl_matrix = nn1.predict(input[i]);
    auto fmatrix = cl_matrix.toFloatMatrix();
    std::cout << input_buffer[i](0, 0) << "|" << input_buffer[i](1, 0) << " = " << fmatrix << "("
              << fmatrix(0, 0) << ")" << std::endl;
  }
}


int main() {
  utils::clWrapper::initOpenCL(*utils::clWrapper::makeDefault());
  runXor(100, 0.1, 0.01);
  return 0;
}