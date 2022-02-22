#pragma once

#include "Optimization.hpp"
#include <vector>


namespace nnet {

  /**
   * @brief RProp optimization
   */
  class RPropPOptimization : public Optimization {
  public:
    explicit RPropPOptimization(const MLPerceptron &perceptron, utils::clWrapper &wrapper,
                                float eta_p = 1.2, float eta_m = 0.5, float lr_max = 50.0,
                                float lr_min = 1e-6);

    void optimize(BackpropStorage &storage, utils::clWrapper &wrapper,
                  cl::CommandQueue &queue) override;

  private:
    std::vector<math::clFMatrix> weights_updates;
    std::vector<math::clFMatrix> old_gradients;
    std::vector<math::clFMatrix> weights_changes;
    const float eta_plus;
    const float eta_minus;
    const float update_max;
    const float update_min;
  };
}   // namespace nnet
