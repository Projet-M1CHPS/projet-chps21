
#include "Network.hpp"
#include "ProjectVersion.hpp"
#include "tscl.hpp"

#include <iomanip>
#include <iostream>
#include <vector>


template<typename T>
void new_func_xor(const size_t bach_size, const T learning_rate, const T error_limit) {
  nnet::MLPerceptron<T> nn1;
  std::vector<size_t> topology = {2, 3, 3, 1};
  nn1.setTopology(topology);
  nn1.setActivationFunction(af::ActivationFunctionType::sigmoid);
  nn1.randomizeWeight();

  nnet::MLPerceptron<T> nn2(nn1);
  nn2.setTopology(topology);
  nn2.setActivationFunction(af::ActivationFunctionType::sigmoid);

  auto &w1 = nn1.getWeights();
  auto &w2 = nn2.getWeights();
  for (size_t i = 0; i < w1.size(); i++) w2[i] = w1[i];

  nnet::SGDOptimization<T> tmStandard(0.2f);
  nnet::MomentumOptimization<T> tmMomentum(topology, 0.1f, 0.9f);

  nnet::MLPModelStochOptimizer<T> opt1(&nn1, &tmStandard);
  nnet::MLPModelStochOptimizer<T> opt2(&nn2, &tmMomentum);

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

int main() {
  xor();
  return 0;
}