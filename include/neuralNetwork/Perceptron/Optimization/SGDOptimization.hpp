#pragma once
#include "Optimization.hpp"

namespace nnet {

  /**
   * @brief Stochastic gradient descent Optimization
   */
  class SGDOptimization : public Optimization {
  public:
    explicit SGDOptimization(const MLPerceptron &perceptron, float lr);

    void optimize(math::clFMatrix &gradient, math::clFMatrix &dest, size_t layer,
                  cl::CommandQueue &queue) override;

  private:
    const float learning_r;
  };
}   // namespace nnet