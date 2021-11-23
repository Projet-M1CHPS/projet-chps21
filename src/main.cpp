#include "ActivationFunction.hpp"
#include "Image.hpp"
#include "Matrix.hpp"
#include "NeuralNetwork.hpp"
#include "Utils.hpp"
#include <array>
#include <iostream>
#include <utility>
#include <vector>
#include <iomanip>

template<typename T>
size_t func_xor(const size_t bach_size, const T learning_rate, const T error_limit) {
  nnet::NeuralNetwork<T> nn;
  nn.setLayersSize(std::vector<size_t>{2, 10,  10, 1});
  nn.setActivationFunction(af::ActivationFunctionType::sigmoid);
  nn.randomizeSynapses();

  std::cout << nn << std::endl;

  std::vector<std::vector<T>> input{{1, 1},
                                    {1, 0},
                                    {0, 1},
                                    {0, 0}};
  std::vector<T> target{0, 1, 1, 0};

  T error = 1.0;
  size_t count = 0;
  while (error > error_limit) {
    for (int i = 0; i < bach_size; i++)
      for (int j = 0; j < 4; j++)
        nn.train(input[j].begin(), input[j].end(), target.begin() + j, target.begin() + j + 1, learning_rate);

    error = 0.0;
    for (int i = 0; i < input.size(); i++)
      error += std::pow(std::fabs(nn.predict(input[i].begin(), input[i].end())(0, 0) - target[i]), 2);
    error /= input.size();
    //std::cout << error << std::endl;
    printf("%.17lf\n", error);
    count++;
  }

  std::cout << "Result"
            << "---> " << count << " iterations" << std::endl;
  for (int i = 0; i < input.size(); i++) {
    std::cout << input[i][0] << "|" << input[i][1] << " = "
              << nn.predict(input[i].begin(), input[i].end()) << "("
              << target[i] << ")" << std::endl;
  }
  //std::cout << nn << std::endl;
  return count;
}

void test() {
  using namespace math;

  Matrix<float> A(2, 2), B(2, 2);
  A(0, 0) = 1;
  A(0, 1) = 2;
  A(1, 0) = 3;
  A(1, 1) = 4;

  B(0, 0) = 4;
  B(0, 1) = 1;
  B(1, 0) = 2;
  B(1, 1) = 6;

  std::cout << A << "\n"
            << B << "\n"
            << B.transpose() << "\n";

  Matrix<float> C = Matrix<float>::MatMatTransProd(A, B);
  std::cout << C << std::endl;
}

int main(int argc, char **argv) {
  func_xor<float>(100, 1.0, 0.001);
  //test();

  return 0;
}