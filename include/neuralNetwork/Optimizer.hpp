#pragma once

#include "math/Matrix.hpp"
#include "Model.hpp"
#include "OptimizerOperation.hpp"
#include <vector>

namespace nnet {

  /**
   * @brief Base class for all optimizers
   */
  class Optimizer {
  public:
    virtual ~Optimizer() = default;

    /**
     * @brief Return an operation that can be used to handle the optimization process. This object
     * should be fed to a OptimizerScheduler.
     * @return A pointer to an OptimizerScheduler.
     */
    virtual std::unique_ptr<OptimizerOperation> makeBatchOperation() = 0;

    /**
     * @brief The optimizer may hold some internal state that needs updating after each epoch
     * This isn't done automatically since the user may repeat the optimization multiple times
     * Before updating
     */
    virtual void update() = 0;
  };

}   // namespace nnet
