#pragma once
#include "Optimization.hpp"

namespace nnet {

  /**
   * @brief SGD variant using a decaying learning rate
   */
  class DecayOptimization : public Optimization {
  public:
    DecayOptimization(const MLPerceptron &perceptron, const float lr_0, const float dr)
        : initial_lr(lr_0), decay_r(dr), learning_r(lr_0), epoch(0) {}

    void optimize(math::clFMatrix &gradient, math::clFMatrix& dest, size_t layer, cl::CommandQueue &queue) override;
    void update() override;

  private:
    const float initial_lr;
    const float decay_r;
    float learning_r;

    size_t epoch = 0;
  };

}   // namespace nnet