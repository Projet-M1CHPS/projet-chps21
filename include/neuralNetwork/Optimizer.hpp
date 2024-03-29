#pragma once

#include "Model.hpp"
#include "math/Matrix.hpp"
#include <vector>

namespace nnet {

  /**
   * @brief Interface class for all optimizers
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
    std::unique_ptr<Operation> makeOperation() {
      auto *ptr = makeOperationImpl();
      return std::unique_ptr<Operation>(ptr);
    }

    /**
     * @brief The optimizer may hold some internal state that needs updating after each epoch
     * This isn't done automatically since the user may repeat the optimization multiple times
     * Before updating
     */
    virtual void update() = 0;

  private:
    virtual Operation *makeOperationImpl() = 0;
  };

  /**
   * @brief High level interface that wraps the Optimizer and provides parallelism
   * capabilities.
   *
   * An optimizer operation receives some inputs, and runs the optimization algorithm on them. Any
   * changes to the model must be cached. At the end of the training epoch, the optimizer operation
   * shall be called to update the model using the cached changes.
   *
   * Multiple thread must be able use the same optimizer operation concurrently, and the optimizer
   * operation may be called multiple times in a row without an update.
   *
   * The operation should apply all the changes that were cached since the last update.
   */
  class Optimizer::Operation {
  public:
    virtual ~Operation() = default;

    /**
     * @brief Run the optimizer on some inputs, using the given device. This operation must be
     * thread-safe and reentrant. If this operation is called multiple times in a row (or
     * concurrently), it must sum any changes to the model until updateModel() is called.
     * @param thread_rank The rank of the thread that is calling this operation. Can be used to
     * use different caches for different threads. Must be in the range [0, num_threads).
     * @param inputs The input for this operation.
     * @param targets The targets corresponding to the inputs.
     * @param batch_device The device that must be used to run this operation.
     */
    virtual void operator()(size_t thread_rank, const math::clFTensor &inputs,
                            const math::clFTensor &targets, cl::CommandQueue queue) = 0;

    /**
     * @brief The number of caches to reserve for this operation. This is used to reserve memory for
     * each thread for local reduction. This value must be in the range [1, num_threads).
     * @param num_threads
     */
    virtual void reserveCaches(size_t num_threads) = 0;

    /**
     * @brief Update the model using the cached changes. This operation is guaranteed to be called a
     * single time, at the end of the epoch.
     *
     * When this method returns, the model update must be completely
     * finished, and the optimizer operation must be ready to be called again.
     * @param queue The queue to use for the model update
     */
    void updateModel(cl::CommandQueue &queue) {
      reduceAll(queue);
      applyChanges(queue);
      clearChanges(queue);
      queue.finish();
    }

  private:
    /**
     * @brief Sums all the changes in the caches
     * @param queue
     */
    virtual void reduceAll(cl::CommandQueue &queue) = 0;

    /**
     * @brief Apply all the changes to the model
     * @param queue
     */
    virtual void applyChanges(cl::CommandQueue &queue) = 0;

    /**
     * @brief Clear all the caches
     * @param queue
     */
    virtual void clearChanges(cl::CommandQueue &queue) = 0;
  };
}   // namespace nnet
