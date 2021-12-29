
#include "Network.hpp"
#include "ProjectVersion.hpp"
#include "tscl.hpp"

#include <iomanip>
#include <iostream>
#include <vector>


void runXor(const size_t bach_size, const float learning_rate, const float error_limit) {
  nnet::MLPModel model;
  auto &nn1 = model.getPerceptron();
  nnet::MLPTopology topology = {2, 3, 3, 1};
  nn1.setTopology(topology);
  nn1.setActivationFunction(af::ActivationFunctionType::sigmoid);
  nn1.randomizeWeight();

  auto &w1 = nn1.getWeights();

  auto tmStandard = std::make_shared<nnet::SGDOptimization<float>>(0.2f);

  nnet::MLPModelStochOptimizer<float> opt1(model, tmStandard);

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
      for (int j = 0; j < 4; j++) { opt1.train(input[j], target[j]); }
    }

    error = 0.0;
    for (int i = 0; i < input.size(); i++) {
      error += std::fabs(nn1.predict(input[i])(0, 0) - target[i](0, 0));
    }
    error /= input.size();
    std::cout << error << std::endl;
    count++;
  }

  std::cout << nn1 << std::endl;
  std::cout << "Result"
            << "---> " << count << " iterations" << std::endl;
  for (int i = 0; i < input.size(); i++) {
    std::cout << input[i](0, 0) << "|" << input[i](1, 0) << " = " << nn1.predict(input[i]) << "("
              << target[i](0, 0) << ")" << std::endl;
  }
}

int main() {
  runXor(100, 0.1, 0.001);
  return 0;
}