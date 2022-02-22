#pragma once
#include "Optimization.hpp"

namespace nnet {

  /**
   * @brief SGD Variant using momentum
   */
  class MomentumOptimization : public Optimization {
  public:
    MomentumOptimization(const MLPerceptron &perceptron, utils::clWrapper &wrapper,
                         float learning_rate, float momentum);

    void optimize(BackpropStorage &storage, utils::clWrapper &wrapper,
                  cl::CommandQueue &queue) override;

  private:
    const float lr;
    const float momentum;
    std::vector<math::clFMatrix> old_weight_change;
  };
}   // namespace nnet