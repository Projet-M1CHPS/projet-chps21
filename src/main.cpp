#include "ActivationFunction.hpp"
#include "NeuralNetwork.hpp"
#include "Utils.hpp"
#include "controlSystem/RunConfiguration.h"
#include "controlSystem/RunControl.h"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace control;

template<typename T>
size_t func_xor(const size_t bach_size, const T learning_rate, const T error_limit) {
  nnet::NeuralNetwork<T> nn;
  nn.setLayersSize(std::vector<size_t>{2, 10,  10, 1});
  nn.setActivationFunction(af::ActivationFunctionType::sigmoid);
  nn.randomizeSynapses();

  std::cout << nn << std::endl;

  std::vector<std::vector<T>> input{{1, 1}, {1, 0}, {0, 1}, {0, 0}};
  std::vector<T> target{0, 1, 1, 0};

  T error = 1.0;
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
    //std::cout << error << std::endl;
    printf("%.17lf\n", error);
    count++;
  }

  std::cout << "Result"
            << "---> " << count << " iterations" << std::endl;
  for (int i = 0; i < input.size(); i++) {
    std::cout << input[i][0] << "|" << input[i][1] << " = "
              << nn.predict(input[i].begin(), input[i].end()) << "(" << target[i] << ")"
              << std::endl;
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
  //func_xor<float>(100, 1.0, 0.001);
  //test();

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