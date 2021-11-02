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
void func_xor(const T learning_rate)
{
  nnet::NeuralNetwork<T> nn;
  nn.setLayersSize(std::vector<size_t>{2, 2, 1});
  nn.setActivationFunction(af::ActivationFunctionType::leakyRelu);
  nn.randomizeSynapses();

  std::cout << nn << std::endl;

  std::vector<std::vector<T>> input{{1, 1},
                                    {1, 0},
                                    {0, 1},
                                    {0, 0}};
  std::vector<T> target{0, 1, 1, 0};

  float error = 1.f;
  size_t count = 0;
  while (error > 0.1)
  {
    for (int i = 0; i < 1000; i++)
      for (int j = 0; j < 4; j++)
      {
        //const size_t n = rand() % 4;
        //nn.train_bis(input[n].begin(), input[n].end(), target.begin() + n, target.begin() + n + 1, learning_rate);
        nn.train_bis(input[j].begin(), input[j].end(), target.begin() + j, target.begin() + j + 1, learning_rate);
      }

    error = 0.0;
    for (int i = 0; i < 4; i++)
      error += std::fabs(nn.forward(input[i].begin(), input[i].end())(0, 0) - target[i]);
    error /= 4;
    std::cout << error << std::endl;
    count++;
  }

  std::cout << "Result" << "---> " << count << std::endl;
  for (int i = 0; i < 4; i++)
  {
    std::cout << input[i][0] << "|" << input[i][1] << " = " << target[i] << std::endl;
    std::cout << nn.forward(input[i].begin(), input[i].end()) << std::endl;
  }
  std::cout << nn << std::endl;
}




int main(int argc, char **argv)
{
  func_xor<float>(0.1);

  return 0;
}