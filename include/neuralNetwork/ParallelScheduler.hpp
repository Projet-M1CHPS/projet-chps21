#pragma once
#include "OptimizationBatchScheduler.hpp"
#include "ParallelPolicy.hpp"

namespace nnet {


  /**
   * @brief A scheduler that parallelize the execution of a neural network. Batches are distributed
   * among all threads, and all OpenCL devices. Use ParallelScheduler::Policy to specify the
   * allocated resources.
   */
  class ParallelScheduler : public OptimizationBatchScheduler {
  public:
    ParallelScheduler(size_t batch_size, Optimizer &optimizer, const ParallelPolicy &policy);

    const utils::ParallelPolicy &getPolicy() const;

    void run() override;

  protected:
    void updateModel() override;
    void print(std::ostream &os) const override;

    void epochStart() override{};
    void endEpoch() override{};

    // We need to allocate the worker_pool on the heap, since it is not copyable and we don't know
    // the number of thread in advance
    std::unique_ptr<boost::asio::thread_pool> worker_pool;
    size_t worker_pool_size;

    const utils::ParallelPolicy policy;
  };

}   // namespace nnet