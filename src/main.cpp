#include "ActivationFunction.hpp"
#include "NeuralNetwork.hpp"
#include "TrainingMethod.hpp"
#include "Utils.hpp"
#include "controlSystem/RunConfiguration.hpp"
#include "controlSystem/RunControl.hpp"
#include <iomanip>
#include <iostream>
#include <vector>


#include "Optimizer.hpp"


// using namespace control;

template<typename T>
size_t func_xor(const size_t bach_size, const T learning_rate, const T error_limit) {
  std::vector<size_t> topology{2, 3, 3, 1};

  nnet::NeuralNetwork<T> nn;
  nn.setLayersSize(topology);
  nn.setActivationFunction(af::ActivationFunctionType::sigmoid);
  nn.randomizeSynapses();

  nnet::StandardTrainingMethod<T> std_mt(learning_rate);
  nnet::MomentumTrainingMethod<T> mom_mt(topology, learning_rate, 0.9);

  nnet::MLPOptimizer<T> opt(&nn, &mom_mt);

  std::cout << nn << std::endl;

  std::vector<math::Matrix<T>> input(4);
  std::vector<math::Matrix<T>> target(4);

  for (size_t i = 0; i < 4; i++) {
    input[i] = math::Matrix<T>(2, 1);
    target[i] = math::Matrix<T>(1, 1);
  }
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

  T error = 1.0;
  size_t count = 0;
  while (error > error_limit) {
    for (int i = 0; i < bach_size; i++) {
      for (int j = 0; j < 4; j++) { opt.train(input[j], target[j]); }
    }

    error = 0.0;
    for (int i = 0; i < input.size(); i++)
      error += std::fabs(nn.predict(input[i])(0, 0) - target[i](0, 0));
    error /= input.size();
    std::cout << std::setprecision(17) << error << std::endl;
    count++;
  }

  std::cout << nn << std::endl;
  std::cout << "Result"
            << "---> " << count << " iterations" << std::endl;
  for (int i = 0; i < input.size(); i++) {
    std::cout << input[i](0, 0) << "|" << input[i](1, 0) << " = " << nn.predict(input[i]) << "("
              << target[i] << ")" << std::endl;
  }
  return count;
}

template<typename T>
void new_func_xor(const size_t bach_size, const T learning_rate, const T error_limit) {
  nnet::NeuralNetwork<T> nn1;
  std::vector<size_t> topology = {2, 3, 3, 1};
  nn1.setLayersSize(topology);
  nn1.setActivationFunction(af::ActivationFunctionType::sigmoid);
  nn1.randomizeSynapses();

  nnet::NeuralNetwork<T> nn2(nn1);
  nn2.setLayersSize(topology);
  nn2.setActivationFunction(af::ActivationFunctionType::sigmoid);

  auto &w1 = nn1.getWeights();
  auto &w2 = nn2.getWeights();
  for (size_t i = 0; i < w1.size(); i++) w2[i] = w1[i];

  nnet::StandardTrainingMethod<T> tmStandard(0.2f);
  nnet::MomentumTrainingMethod<T> tmMomentum(topology, 0.1f, 0.9f);

  nnet::MLPOptimizer<T> opt1(&nn1, &tmStandard);
  nnet::MLPOptimizer<T> opt2(&nn2, &tmMomentum);

  std::cout << nn1 << std::endl;
  std::cout << nn2 << std::endl;

  std::vector<math::Matrix<T>> input(4);
  std::vector<math::Matrix<T>> target(4);
  for (size_t i = 0; i < 4; i++) {
    input[i] = math::Matrix<T>(2, 1);
    target[i] = math::Matrix<T>(1, 1);
  }
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

  for (int z = 0; z < 2; z++) {
    T error = 1.0;
    size_t count = 0;
    while (error > error_limit && count < 600) {
      for (int i = 0; i < bach_size; i++) {
        for (int j = 0; j < 4; j++) {
          if (z == 0) {
            opt1.train(input[j], target[j]);
          } else {
            opt2.train(input[j], target[j]);
          }
        }
      }

      error = 0.0;
      for (int i = 0; i < input.size(); i++) {
        if (z == 0) error += std::fabs(nn1.predict(input[i])(0, 0) - target[i](0, 0));
        else
          error += std::fabs(nn2.predict(input[i])(0, 0) - target[i](0, 0));
      }
      error /= input.size();
      std::cout << std::setprecision(17) << error << std::endl;
      count++;
    }

    if (z == 0) {
      std::cout << nn1 << std::endl;
      std::cout << "Result"
                << "---> " << count << " iterations" << std::endl;
      for (int i = 0; i < input.size(); i++) {
        std::cout << input[i](0, 0) << "|" << input[i](1, 0) << " = " << nn1.predict(input[i])
                  << "(" << target[i](0, 0) << ")" << std::endl;
      }
    } else {
      std::cout << nn2 << std::endl;
      std::cout << "Result"
                << "---> " << count << " iterations" << std::endl;
      for (int i = 0; i < input.size(); i++) {
        std::cout << input[i](0, 0) << "|" << input[i](1, 0) << " = " << nn2.predict(input[i])
                  << "(" << target[i](0, 0) << ")" << std::endl;
      }
    }
  }
}


void test();


int main(int argc, char **argv) {
  // func_xor<float>(100, 0.2, 0.01);
  //new_func_xor<float>(100, 0.2, 0.000001);
  // test();




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

  control::RunConfiguration config(argv[1], working_dir, target_dir);
  auto controller = std::make_unique<control::TrainingRunController>();

  control::RunResult res = controller->launch(config);
  controller->cleanup();

  if (not res) {
    std::cerr << "Run failed: " << res.getMessage() << std::endl;
    return 1;
  }

  return 0;
}