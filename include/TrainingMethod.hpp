#pragma once

#include "Matrix.hpp"
#include "NeuralNetwork.hpp"
#include <vector>


namespace nnet {
  template<typename T>
  class NeuralNetwork;

  template<typename T>
  class StandardTrainingMethod {
  public:
    StandardTrainingMethod(const T learningRate) : learningRate(learningRate){};
    virtual ~StandardTrainingMethod() = default;

    void compute(math::Matrix<T> &weights, const math::Matrix<T> &gradient) {
      weights -= (gradient * learningRate);
    }

  private:
    const T learningRate;
  };


  template<typename T>
  class MomentumTrainingMethod {
  public:
    MomentumTrainingMethod(const std::vector<size_t> &topology, const T learningRate,
                           const T momentum)
        : learningRate(learningRate), momentum(momentum) {
      for (size_t i = 0; i < topology.size() - 1; i++) {
        dw_old.push_back(math::Matrix<T>(topology[i + 1], topology[i]));
      }
    };
    virtual ~MomentumTrainingMethod() = default;

    void compute(const size_t index, math::Matrix<T> &weights, const math::Matrix<T> &gradient) {
      auto dw = (gradient * learningRate) + (dw_old[index] * momentum);
      weights -= dw;
      dw_old[index] = std::move(dw);
    }

  private:
    T learningRate;
    T momentum;
    std::vector<math::Matrix<T>> dw_old;
  };


}   // namespace nnet
