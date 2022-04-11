#pragma once
#include "OptimizationScheduler.hpp"
#include "Optimizer.hpp"
#include "math/clFTensor.hpp"

namespace nnet {

  /**
   * @brief Interface for a scheduler that optimizes the weights of the network using batches of
   * data. Batches are distributed among the available resources.
   */
  class BatchOptimizationScheduler : public OptimizationScheduler {
  public:
    /**
     * @brief Build a new BatchOptimizationScheduler.
     * @param batch_size The number of inputs to batch together for each optimization.
     * @param inputs A list of tensors that are the inputs to the network.
     * @param targets The targets corresponding to the inputs. The number of targets must be equal
     * to the number of inputs, and tensors must have the same shape. The tensors must remain alive
     * for the duration of the optimization, and are not copied.
     */
    BatchOptimizationScheduler(size_t batch_size, Optimizer &optimizer,
                               const std::vector<math::clFTensor> &inputs,
                               const std::vector<math::clFTensor> &targets);

    /**
     * @brief Get the size of the batches that will be used for the optimization.
     * @return
     */
    size_t getBatchSize() const { return batch_size; }

    /**
     * @brief Set the size of the batches that will be used for the optimization.
     * @param new_batch_size
     */
    void setBatchSize(size_t new_batch_size) { batch_size = new_batch_size; }

  protected:
    size_t batch_size;
    const std::vector<math::clFTensor> *const input_tensors;
    const std::vector<math::clFTensor> *const target_tensors;
    const std::unique_ptr<Optimizer::Operation> optimizer_operation;
    Optimizer *optimizer;
  };
}   // namespace nnet
