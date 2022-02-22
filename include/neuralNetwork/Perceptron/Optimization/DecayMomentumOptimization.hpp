#pragma once
#include "Optimization.hpp"

namespace nnet {

  /**
   * @brief Combined Decay and Momentum optimization
   */
  class DecayMomentumOptimization : public Optimization {
  public:
    DecayMomentumOptimization(const MLPerceptron &perceptron, utils::clWrapper &wrapper, float lr_0,
                              float dr, float mom);

    void optimize(BackpropStorage &storage, utils::clWrapper &wrapper,
                  cl::CommandQueue &queue) override;
    void update() override;

  private:
    const float initial_lr;
    const float decay_r;
    float learning_r;
    const float momentum;

    size_t epoch = 0;

    std::vector<math::clFMatrix> old_weight_change;
  };

}   // namespace nnet