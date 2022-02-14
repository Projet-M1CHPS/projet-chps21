#pragma once
#include "Optimization.hpp"

namespace nnet {

  /**
   * @brief Stochastic gradient descent optimization without momentum
   */
  class SGDOptimization : public Optimization {
  public:
    explicit SGDOptimization(const MLPerceptron &perceptron, float lr);

    void optimize(BackpropStorage &storage) override;

  private:
    const float learning_r;
  };
}   // namespace nnet