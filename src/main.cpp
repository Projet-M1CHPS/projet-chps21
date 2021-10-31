#include "Image.hpp"
#include "NeuralNetwork.hpp"
#include "Utils.hpp"
#include "Matrix.hpp"
#include "ActivationFunction.hpp"
#include <iostream>
#include <vector>
#include <utility>
#include <array>

template <typename T>
void func_xor()
{
  nnet::NeuralNetwork<T> nn;
  nn.setLayersSize(std::vector<size_t>{2, 2, 1});
  nn.setActivationFunction(af::ActivationFunctionType::sigmoid);
  nn.randomizeSynapses();

  auto &b = nn.getBiases();

  for (auto &i : b)
    for (auto &e : i)
      e = 0;

  std::cout << nn << std::endl;

  std::vector<std::vector<T>> input{{1, 1},
                                    {1, 0},
                                    {0, 1},
                                    {0, 0}};
  std::vector<T> target{0, 1, 1, 0};

  float error = 1.f;
  size_t count = 0;
  while (error > 0.02)
  {
    for (int i = 0; i < 10000; i++)
      for (int j = 0; j < 4; j++)
        nn.train_bis(input[j].begin(), input[j].end(), target.begin() + j, target.begin() + j + 1, 0.075);

    error = 0.0;
    for (int i = 0; i < 4; i++)
      error += std::fabs(nn.forward(input[i].begin(), input[i].end())(0, 0) - target[i]);
    error /= 4;
    std::cout << error << std::endl;
    count++;
  }

  std::cout << "Result" << "--->" << count << std::endl;
  for (int i = 0; i < 4; i++)
  {
    std::cout << input[i][0] << "|" << input[i][1] << " = " << target[i] << std::endl;
    std::cout << nn.forward(input[i].begin(), input[i].end()) << std::endl;
  }
  std::cout << nn << std::endl;
}

// Test main, should be replaced by the python interface
int main(int argc, char **argv)
{
  nnet::NeuralNetwork<float> nn;
  nn.setLayersSize(std::vector<size_t>{2, 2, 1});
  nn.setActivationFunction(af::ActivationFunctionType::sigmoid);
  nn.randomizeSynapses();
  //nn.setActivationFunction(af::ActivationFunctionType::identity, 2);

  auto &w = nn.getWeights();
  auto &b = nn.getBiases();

  for (auto &i : b)
  {
    for (auto &e : i)
      e = 0;
    //std::cout << i << std::endl;
  }

  std::vector<float> input{1, 1};
  std::vector<float> target{-6.255, 20003};

  func_xor<float>();
  //printf("--------------------\n");
  //func_xor<double>();

  return 0;
}