#pragma once
#include "OptimizationScheduler.hpp"
#include "Optimizer.hpp"
#include "math/clFTensor.hpp"

namespace nnet {

  class BatchSchedulerJob {
  public:
    BatchSchedulerJob() = default;
    BatchSchedulerJob(size_t batch_size, const std::vector<math::clFTensor> &inputs,
                      const std::vector<math::clFTensor> &targets);

    size_t getBatchSize() const { return batch_size; }
    const std::vector<math::clFTensor> &getInputs() const { return *inputs; }
    const std::vector<math::clFTensor> &getTargets() const { return *targets; }

    bool isValid() const { return batch_size > 0 && inputs && targets; }
    size_t getGlobalWorkSize() const;

  private:
    size_t batch_size = 0;
    const std::vector<math::clFTensor> *inputs = nullptr;
    const std::vector<math::clFTensor> *targets = nullptr;
  };

  /**
   * @brief Interface for a scheduler that optimizes the weights of the network using batches of
   * data. Batches are distributed among the available resources.
   */
  class BatchOptimizationScheduler : public OptimizationScheduler {
  public:
    /**
     * @brief Build a new BatchOptimizationScheduler.
     * @param job
     */
    explicit BatchOptimizationScheduler(const BatchSchedulerJob &job);

    /**
     * @brief Get the size of the batches that will be used for the optimization.
     * @return
     */
    const BatchSchedulerJob &getJob() const { return job; }

    /**
     * @brief Set the size of the batches that will be used for the optimization.
     * @param new_batch_size
     */
    void setJob(const BatchSchedulerJob &new_job) { this->job = new_job; }

  private:
    BatchSchedulerJob job;
  };
}   // namespace nnet
