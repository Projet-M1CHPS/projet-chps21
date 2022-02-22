#pragma once
#include "Optimization.hpp"

namespace nnet {

  /**
   * @brief Stochastic gradient descent optimization without momentum
   */
  class SGDOptimization : public Optimization {
  public:
    explicit SGDOptimization(const MLPerceptron &perceptron, utils::clWrapper &wrapper, float lr);

    void optimize(BackpropStorage &storage, utils::clWrapper &wrapper,
                  cl::CommandQueue &queue) override;

  private:
    const float learning_r;
  };
}   // namespace nnet