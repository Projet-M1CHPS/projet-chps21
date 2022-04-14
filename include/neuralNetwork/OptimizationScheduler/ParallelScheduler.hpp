#pragma once
#include "BatchOptimizationScheduler.hpp"
#include "BatchProgression.hpp"

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
    class Builder;

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

    ParallelScheduler(const ParallelScheduler &other) = delete;
    ParallelScheduler(ParallelScheduler &&other) = default;

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
    virtual ~Dispatcher() = default;
    virtual void dispatch(BatchProgression &progression, size_t count,
                          Optimizer::Operation &op) = 0;
  };

  /**
   * @brief Describes the resources allocated to the parallel scheduler. Lightweight class.
   */
  class ParallelScheduler::Policy {
  public:
    Policy(size_t max_thread, bool multiple_thread_per_device, std::vector<cl::Device> devices);

    size_t getMaxThread() const { return max_thread; }

    bool hasMultipleThreadPerDevice() const { return multiple_thread_per_device; }

    const std::vector<cl::Device> &getDevices() const { return devices; }

  private:
    size_t max_thread;
    bool multiple_thread_per_device;
    std::vector<cl::Device> devices;
  };

  class ParallelScheduler::Builder {
  public:
    void setTrainingSets(const std::vector<math::clFTensor> &inputs,
                         const std::vector<math::clFTensor> &targets) {
      this->inputs = &inputs;
      this->targets = &targets;
    }
    // Set the resources for the scheduler
    void setMaxThread(size_t nmax_thread, bool nallow_multiple_thread_per_device) {
      max_thread = nmax_thread;
      multiple_thread_per_device = nallow_multiple_thread_per_device;
    }
    void setDevices(const std::vector<cl::Device> &ndevices) { devices = ndevices; }

    // Set the batch size, and allow/disallow batch optimization
    void setBatchSize(size_t nbatch_size, bool nallow_batch_defragmentation) {
      batch_size = nbatch_size;
      allow_batch_defragmentation = nallow_batch_defragmentation;
    }

    void setOptimizer(Optimizer &optimizer) { this->optimizer = &optimizer; }

    std::unique_ptr<ParallelScheduler> build() const {
      if (not inputs or not targets or not optimizer or devices.empty()) {
        throw std::runtime_error("ParallelScheduler::Builder: not all required parameters are set");
      }
      Policy policy(max_thread, multiple_thread_per_device, devices);
      auto scheduler = ParallelScheduler::makeDefaultDispatcher(*inputs, *targets, batch_size,
                                                                *optimizer, policy);
      return std::make_unique<ParallelScheduler>(std::move(scheduler));
    }

  private:
    const std::vector<math::clFTensor> *inputs, *targets;

    Optimizer *optimizer;

    size_t max_thread = 1;
    bool multiple_thread_per_device = false;
    std::vector<cl::Device> devices = utils::cl_wrapper.getDevices();

    size_t batch_size = 1;
    bool allow_batch_defragmentation = false;
  };

}   // namespace nnet