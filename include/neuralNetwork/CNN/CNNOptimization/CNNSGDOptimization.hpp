#pragma once

#include "CNN.hpp"
#include "CNNOptimization.hpp"
#include "clFTensor.hpp"


namespace nnet {

  /**
   * @brief Stochastic gradient descent optimization without momentum for CNN
   */
  class CNNSGDOptimization : public CNNOptimization {
  public:
    explicit CNNSGDOptimization(const CNN &cnn, float lr) : learning_r(lr) {}

    void optimize(const math::clFTensor &gradient, math::clFTensor &dest,
                  cl::CommandQueue &queue) override;

  private:
    const float learning_r;
  };

}   // namespace nnet