#pragma once
#include "Optimization.hpp"

namespace nnet {

  /**
   * @brief SGD Variant using momentum
   */
  class MomentumOptimization : public Optimization {
  public:
    MomentumOptimization(const MLPerceptron &perceptron, float learning_rate, float momentum);

    void optimize(BackpropStorage &storage) override;

  private:
    const float lr;
    const float momentum;
    std::vector<math::FloatMatrix> old_weight_change;
  };
}   // namespace nnet