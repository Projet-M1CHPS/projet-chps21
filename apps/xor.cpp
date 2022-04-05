
#include "NeuralNetwork.hpp"
#include "ProjectVersion.hpp"
#include "tscl.hpp"

#include <iomanip>
#include <iostream>
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

  std::vector<math::FloatMatrix> input(4);
  std::vector<math::FloatMatrix> target(4);
  for (size_t i = 0; i < 4; i++) {
    input[i] = math::FloatMatrix(2, 1);
    target[i] = math::FloatMatrix(1, 1);
  }

  // Xor truth table
  // Yes, this is ugly, but who cares ?
  input[0](0, 0) = 0.f;
  input[0](1, 0) = 0.f;
  input[1](0, 0) = 1.f;
  input[1](1, 0) = 0.f;
  input[2](0, 0) = 0.f;
  input[2](1, 0) = 1.f;
  input[3](0, 0) = 1.f;
  input[3](1, 0) = 1.f;

  target[0](0, 0) = 0.f;
  target[1](0, 0) = 1.f;
  target[2](0, 0) = 1.f;
  target[3](0, 0) = 0.f;

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
      error += std::fabs(fmatrix(0, 0) - target[i](0, 0));
    }
    error /= input.size();
    std::cout << error << std::endl;
    count++;
  }

  std::cout << nn1 << std::endl;
  std::cout << "Result"
            << "---> " << count << " iterations" << std::endl;
  for (int i = 0; i < input.size(); i++) {
    auto cl_matrix = nn1.predict(input[i]);
    auto fmatrix = cl_matrix.toFloatMatrix();
    std::cout << input[i](0, 0) << "|" << input[i](1, 0) << " = " << fmatrix << "("
              << target[i](0, 0) << ")" << std::endl;
  }
}

int main() {
  utils::clWrapper::initOpenCL(*utils::clWrapper::makeDefault());
  runXor(100, 0.1, 0.01);
  return 0;
}