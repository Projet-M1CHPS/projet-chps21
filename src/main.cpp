#include "Image.hpp"
#include "NeuralNetwork.hpp"
#include "Utils.hpp"
#include "Matrix.hpp"
#include "ActivationFunction.hpp"
#include <iostream>
#include <vector>

// Test main, should be replaced by the python interface
int main(int argc, char **argv)
{
  ///////Ugo
  nnet::NeuralNetwork<float> nn;
  nn.setLayersSize(std::vector<size_t>{2, 2, 1});
  nn.setActivationFunction(af::ActivationFunctionType::sigmoid);
  nn.randomizeSynapses();

  auto &w = nn.getWeights();
  auto &b = nn.getBiases();

  for (auto &i : w)
  {
    //for (auto &e : i)
      //e = ((float)rand()/MAXFLOAT) + 0.01;
      //e = 1.f;
    std::cout << i << std::endl;
  }

  for (auto &i : b)
  {
    //for (auto &e : i)
    //  e = 0;
    std::cout << i << std::endl;
  }

  std::vector<float> input{1, 1};
  std::vector<float> target{0};
  

  std::cout << "forward :\n" << nn.forward(input.begin(), input.end());
  for(int i = 0; i < 1000000; i++)
    nn.train(input.begin(), input.end(), target.begin(), target.end());

  std::cout << "forward :\n" << nn.forward(input.begin(), input.end());

  /*for(int i = 0; i < 1000000; i++)
    nn.train(input.begin(), input.end(), target.begin(), target.end());

  std::cout << "forward :\n" << nn.forward(input.begin(), input.end()); */

  return 0;
}