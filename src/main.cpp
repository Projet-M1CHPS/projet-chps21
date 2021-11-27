#include "ActivationFunction.hpp"
#include "NeuralNetwork.hpp"
#include "Utils.hpp"
#include "controlSystem/RunConfiguration.hpp"
#include "controlSystem/RunControl.hpp"
#include <iomanip>


#include <iostream>
#include <vector>

using namespace control;

template<typename T>
size_t func_xor(const size_t bach_size, const T learning_rate, const T error_limit) {
  nnet::NeuralNetwork<T> nn;
  nn.setLayersSize(std::vector<size_t>{2, 3, 3, 1});
  nn.setActivationFunction(af::ActivationFunctionType::sigmoid);
  // nn.setActivationFunction(af::ActivationFunctionType::sigmoid, 2);
  nn.randomizeSynapses();

  std::cout << nn << std::endl;

  std::vector<std::vector<T>> input{{1, 1}, {1, 0}, {0, 1}, {0, 0}};
  std::vector<T> target{0, 1, 1, 0};

  T error = 1.0;
  size_t count = 0;
  while (error > error_limit) {
    for (int i = 0; i < bach_size; i++) {
      for (int j = 0; j < 4; j++) {
        nn.train(input[j].begin(), input[j].end(), target.begin() + j, target.begin() + j + 1,
                 learning_rate);
      }
    }

    error = 0.0;
    for (int i = 0; i < input.size(); i++)
      error += std::fabs(nn.predict(input[i].begin(), input[i].end())(0, 0) - target[i]);
    error /= input.size();
    std::cout << std::setprecision(17) << error << std::endl;
    count++;
  }

  std::cout << nn << std::endl;
  std::cout << "Result"
            << "---> " << count << " iterations" << std::endl;
  for (int i = 0; i < input.size(); i++) {
    std::cout << input[i][0] << "|" << input[i][1] << " = "
              << nn.predict(input[i].begin(), input[i].end()) << "(" << target[i] << ")"
              << std::endl;
  }
  return count;
}

template<typename T>
void test_matTransMatProd() {
  using namespace math;

  Matrix<T> A(3, 2), B(3, 2);
  A(0, 0) = 1;
  A(0, 1) = 2;
  A(1, 0) = 3;
  A(1, 1) = 4;
  A(2, 0) = 12;
  A(2, 1) = 8;

  B(0, 0) = 4;
  B(0, 1) = 1;
  B(1, 0) = 2;
  B(1, 1) = 6;
  B(2, 0) = 12;
  B(2, 1) = 9;

  std::cout << A.transpose() << "\n" << B << "\n";

  Matrix<T> C = Matrix<T>::matTransMatProd(A, B);
  auto D = Matrix<T>::matMatProd(true, A, false, B);
  std::cout << C << "\n" << D << std::endl;
}

template<typename T>
void test_matMatTransProd() {
  using namespace math;

  Matrix<T> A(3, 2), B(3, 2);
  A(0, 0) = 1;
  A(0, 1) = 2;
  A(1, 0) = 3;
  A(1, 1) = 4;
  A(2, 0) = 12;
  A(2, 1) = 8;

  B(0, 0) = 4;
  B(0, 1) = 1;
  B(1, 0) = 2;
  B(1, 1) = 6;
  B(2, 0) = 12;
  B(2, 1) = 9;

  std::cout << A << "\n" << B.transpose() << "\n";

  Matrix<T> C = Matrix<T>::matMatTransProd(A, B);
  auto D = Matrix<T>::matMatProd(false, A, true, B);
  std::cout << C << "\n" << D << std::endl;
}

template<typename T>
void test_matMatProd() {
  using namespace math;

  Matrix<T> A(3, 2), B(2, 3);
  A(0, 0) = 1;
  A(0, 1) = 2;
  A(1, 0) = 3;
  A(1, 1) = 4;
  A(2, 0) = 12;
  A(2, 1) = 8;

  B(0, 0) = 4;
  B(0, 1) = 1;
  B(0, 2) = 12;
  B(1, 0) = 2;
  B(1, 1) = 6;
  B(1, 2) = 9;

  Matrix<float> C = Matrix<float>::matMatTransProd(A, false, 1.0f, B, true);
  std::cout << C << std::endl;
}

void test_neural_network() {
  nnet::NeuralNetwork<float> nn;
  nn.setLayersSize(std::vector<size_t>{2, 2, 2});
  nn.setActivationFunction(af::ActivationFunctionType::sigmoid);

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

  std::cout << nn << std::endl;

  auto input = math::Matrix<float>(2, 1);
  input(0, 0) = 0.05;
  input(1, 0) = 0.10;
  auto output = math::Matrix<float>(2, 1);
  output(0, 0) = 0.01;
  output(1, 0) = 0.99;

  std::cout << "prediction : \n" << nn.predict(input) << std::endl;

  nn.train(input, output, 0.5);
  std::cout << nn << std::endl;
}


int main(int argc, char **argv) {
  //func_xor<float>(100, 0.2, 0.001);
  // test();
  // test_neural_network();

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <input_dir> (<working_dir>) (<target_dir>)";
    return 1;
  }
  std::string working_dir, target_dir;

  if (argc == 3) working_dir = argv[2];
  else
    working_dir = "runs";

  if (argc >= 4) target_dir = argv[3];
  else
    target_dir = "run_" + utils::timestampAsStr();

  RunConfiguration config(argv[1], working_dir, target_dir);
  auto controller = std::make_unique<TrainingRunController>();

  RunResult res = controller->launch(config);
  controller->cleanup();

  if (not res) {
    std::cerr << "Run failed: " << res.getMessage() << std::endl;
    return 1;
  }

  return 0;
}