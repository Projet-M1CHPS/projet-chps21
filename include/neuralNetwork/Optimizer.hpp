#pragma once

#include "Model.hpp"
#include "math/Matrix.hpp"
#include <vector>

namespace nnet {

  /**
   * @brief Base class for all optimizers
   */
  class Optimizer {
  public:
    class Operation;

    virtual ~Optimizer() = default;

    /**
     * @brief Return an operation that can be used to handle the optimization process. This object
     * should be fed to a OptimizationScheduler.
     * @return A pointer to an OptimizationScheduler.
     */
    virtual std::unique_ptr<Operation> makeBatchOperation() = 0;

    /**
     * @brief The optimizer may hold some internal state that needs updating after each epoch
     * This isn't done automatically since the user may repeat the optimization multiple times
     * Before updating
     */
    virtual void update() = 0;
  };

  /**
   * @brief Interface for an optimizer operation. An optimizer operation receives some inputs, and
   * must run the optimization algorithm on them. Any changes to the model must be cached. At the
   * end of the training epoch, the optimizer operation shall be called to update the model using
   * the cached changes.
   *
   * Multiple thread may use the same optimizer operation concurrently, and the optimizer operation
   * may be called multiple times in a row. The updateModel() operation should apply all the
   * changes that were cached since the last call.
   */
  class Optimizer::Operation {
  public:
    virtual ~Operation() = default;

    /**
     * @brief Run the optimizer on some inputs, using the given device. This operation must be
     * thread-safe and reentrant. If this operation is called multiple times in a row (or
     * concurrently), it must sum any changes to the model until updateModel() is called.
     * @param inputs The input for this operation.
     * @param targets The targets corresponding to the inputs.
     * @param batch_device The device that must be used to run this operation.
     */
    virtual void operator()(const math::clFTensor &inputs, const math::clFTensor &targets,
                            cl::Device &batch_device) = 0;

    /**
     * @brief Update the model using the cached changes. This operation is guaranteed to be called a
     * single time, at the end of the epoch.
     *
     * When this method returns, the model update must be
     * finished, and the optimizer operation must be ready to be called again.
     */
    virtual void updateModel() = 0;
  };

}   // namespace nnet
