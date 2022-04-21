#pragma once

#include "clFTensor.hpp"

namespace nnet {

  /**
   * @brief Base class for all CNN optimization
   */
  class CNNOptimization {
  public:
    CNNOptimization() = default;
    virtual ~CNNOptimization() = default;

    /**
     *
     * @param storage
     */
    virtual void optimize(const math::clFTensor &gradient, math::clFTensor &dest,
                          cl::CommandQueue &queue) = 0;

    /**
     * @brief Some optimization require to be updated after each epoch
     * We provide an empty declaration here to avoid to have to implement it where not needed
     */
    virtual void update(){};
  };

}   // namespace nnet