#pragma once
#include "BatchProgression.hpp"
#include "BatchOptimizationScheduler.hpp"

namespace nnet {

  /**
   * @brief A scheduler that parallelize the execution of a neural network. Batches are distributed
   * among all threads, and all OpenCL devices. Use ParallelScheduler::Policy to specify the
   * allocated resources.
   */
  class ParallelScheduler : public BatchOptimizationScheduler {
  public:
    class Policy;
    class Dispatcher;

    /**
     * @brief Construct a new Parallel Scheduler an takes ownership of a dispatcher
     * @param inputs The inputs for the optimization
     * @param targets The corresponding targets
     * @param batch_size The size of the batches
     * @param optimizer The optimizer to schedule
     * @param dispatcher The dispatcher to use for batch parallelization
     */
    ParallelScheduler(const std::vector<math::clFTensor> &inputs,
                      const std::vector<math::clFTensor> &targets, size_t batch_size,
                      Optimizer &optimizer, std::unique_ptr<Dispatcher> dispatcher);

    /**
     * @brief Construct a new Parallel Scheduler using a default dispatcher
     * @param inputs The inputs for the optimization
     * @param targets The corresponding targets
     * @param batch_size The size of the batches
     * @param optimizer The optimizer to schedule
     * @param policy The policy for the default dispatcher
     * @return
     */
    static ParallelScheduler makeDefaultDispatcher(const std::vector<math::clFTensor> &inputs,
                                                   const std::vector<math::clFTensor> &targets,
                                                   size_t batch_size, Optimizer &optimizer,
                                                   const Policy &policy);

    void run() override;

  protected:
    void updateModel() override;
    void print(std::ostream &os) const override;

    void epochStart() override;
    void endEpoch() override;

    std::unique_ptr<Dispatcher> batch_dispatcher;
  };

  class ParallelScheduler::Dispatcher {
  public:
    virtual void dispatch(BatchProgression &progression, size_t count,
                          Optimizer::Operation &op) = 0;
  };

  /**
   * @brief Describes the resources allocated to the parallel scheduler. Lightweight class.
   */
  class ParallelScheduler::Policy {
  public:
    Policy(size_t max_thread, bool multiple_thread_per_device,
           std::vector<cl::Device> devices);

    size_t getMaxThread() const { return max_thread; }

    bool hasMultipleThreadPerDevice() const { return multiple_thread_per_device; }

    const std::vector<cl::Device> &getDevices() const { return devices; }

  private:

    size_t max_thread;
    bool multiple_thread_per_device;
    std::vector<cl::Device> devices;
  };

}   // namespace nnet