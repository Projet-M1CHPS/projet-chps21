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
    StandardTrainingMethod(const T learningRate) : learningRate(learningRate) {}
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
    }
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


  template<typename T>
  class RproppTrainingMethod {
  public:
    RproppTrainingMethod(const std::vector<size_t> &topology) {
      for (size_t i = 0; i < topology.size() - 1; i++) {
        lr_old.push_back(math::Matrix<T>(topology[i + 1], topology[i]));
        gradient_old.push_back(math::Matrix<T>(topology[i + 1], topology[i]));
      }
      for (auto &&i : lr_old) { i.fill(0.1); }
      for (auto &&i : gradient_old) { i.fill(1.0); }
    }
    virtual ~RproppTrainingMethod() = default;

    void compute(const size_t index, math::Matrix<T> &weights, const math::Matrix<T> &gradient) {
      for (size_t i = 0; i < weights.getRows(); i++) {
        for (size_t j = 0; j < weights.getCols(); j++) {
          T dw = 0.0;
          if (gradient(i, j) * gradient_old[index](i, j) > 0.0) {
            lr_old[index](i, j) = std::min(eta_plus * lr_old[index](i, j), lr_max);
            if (gradient(i, j) > 0) dw = -1 * lr_old[index](i, j);
            else
              dw = lr_old[index](i, j);
            weights(i, j) = weights(i, j) + dw;
          } else if (gradient(i, j) * gradient_old[index](i, j) < 0.0) {
            if (gradient_old[index](i, j) > 0) dw = -1 * lr_old[index](i, j);
            else
              dw = lr_old[index](i, j);
            weights(i, j) = weights(i, j) - dw;
            lr_old[index](i, j) = std::max(eta_minus * lr_old[index](i, j), lr_min);
            gradient(i, j) = 0.0;
          } else {
            if (gradient(i, j) > 0) dw = -1 * lr_old[index](i, j);
            else
              dw = lr_old[index](i, j);
            weights(i, j) = weights(i, j) + dw;
          }
          gradient_old[index](i, j) = gradient(i, j);
        }
      }
    }

  private:
    std::vector<math::Matrix<T>> lr_old;
    std::vector<math::Matrix<T>> gradient_old;
    const T eta_plus = 1.2;
    const T eta_minus = 0.5;
    const T lr_max = 50.0;
    const T lr_min = 0.000001;
  };


}   // namespace nnet
