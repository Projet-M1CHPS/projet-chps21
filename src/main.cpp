#include "ActivationFunction.hpp"
#include "NeuralNetwork.hpp"
#include "RunControl.h"
#include "Utils.hpp"
#include <iostream>
#include <vector>

template<typename T>
size_t func_xor(const size_t bach_size, const T learning_rate, const T error_limit) {
  nnet::NeuralNetwork<T> nn;
  nn.setLayersSize(std::vector<size_t>{2, 2, 2, 1});
  nn.setActivationFunction(af::ActivationFunctionType::leakyRelu);
  nn.randomizeSynapses();

  std::cout << nn << std::endl;

  std::vector<std::vector<T>> input{{1, 1}, {1, 0}, {0, 1}, {0, 0}};
  std::vector<T> target{0, 1, 1, 0};

  float error = 1.f;
  size_t count = 0;
  while (error > error_limit) {
    for (int i = 0; i < bach_size; i++)
      for (int j = 0; j < 4; j++)
        nn.train(input[j].begin(), input[j].end(), target.begin() + j, target.begin() + j + 1,
                 learning_rate);

    error = 0.0;
    for (int i = 0; i < input.size(); i++)
      error += std::pow(std::fabs(nn.predict(input[i].begin(), input[i].end())(0, 0) - target[i]),
                        2);
    error /= input.size();
    std::cout << error << std::endl;
    count++;
  }

  std::cout << "Result"
            << "---> " << count << " iterations" << std::endl;
  for (int i = 0; i < input.size(); i++) {
    std::cout << input[i][0] << "|" << input[i][1] << " = "
              << nn.predict(input[i].begin(), input[i].end()) << "(" << target[i] << ")"
              << std::endl;
  }
  std::cout << nn << std::endl;
  return count;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: ./" << argv[0] << " <input_dir> <working_dir>";
    return 1;
  }

  RunConfiguration config;


  return runOnConfig(config) ? 0 : 1;
}