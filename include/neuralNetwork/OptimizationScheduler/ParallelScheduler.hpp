#pragma once
#include "BatchOptimizationScheduler.hpp"
#include "BatchProgression.hpp"

namespace nnet {

  /**
   * @brief A scheduler that parallelize the training of a neural network. Batches are distributed
   * among all threads, and all OpenCL devices. Use ParallelScheduler::Policy to specify which
   * resources are allowed.
   */
  class ParallelScheduler : public BatchOptimizationScheduler {
  public:
    class Policy;
    class Dispatcher;
    class Builder;

    /**
     * @brief Construct a new Parallel Scheduler an takes ownership of a dispatcher
     * @param job The jobs that needs scheduling
     * @param optimizer The optimizer to schedule
     * @param dispatcher The dispatcher to use for batch parallelization
     */
    ParallelScheduler(const BatchSchedulerJob &job, Optimizer &optimizer,
                      std::unique_ptr<Dispatcher> dispatcher);

    ParallelScheduler(const ParallelScheduler &other) = delete;
    ParallelScheduler(ParallelScheduler &&other) = default;

    /**
     * @brief Construct a new Parallel Scheduler using a default dispatcher
     * @param Job The job that needs scheduling
     * @param optimizer The optimizer to schedule
     * @param policy The policy for the default dispatcher
     * @return
     */
    static ParallelScheduler makeWithDefaultDispatcher(const BatchSchedulerJob &job,
                                                       Optimizer &optimizer, const Policy &policy);

    /**
     * @brief Runs the optimizer for a single epoch. How batch are dispatched is determined by the
     * policy used during the construction of the scheduler.
     */
    void run() override;

  protected:
    void print(std::ostream &os) const override;

    void updateModel() override;
    void epochStart() override;
    void endEpoch() override;

  private:
    std::unique_ptr<Dispatcher> batch_dispatcher;
    Optimizer *optimizer;
    std::unique_ptr<Optimizer::Operation> optimizer_operation;
  };

  class ParallelScheduler::Dispatcher {
  public:
    virtual ~Dispatcher() = default;
    virtual void dispatch(BatchProgression &progression, size_t work_size,
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
    void setJob(const BatchSchedulerJob &njob) { job = njob; }

    // Set the resources for the scheduler
    void setMaxThread(size_t nmax_thread, bool nallow_multiple_thread_per_device) {
      max_thread = nmax_thread;
      multiple_thread_per_device = nallow_multiple_thread_per_device;
    }

    void setDevices(const std::vector<cl::Device> &ndevices) { devices = ndevices; }

    void setOptimizer(Optimizer &new_optimizer) { this->optimizer = &new_optimizer; }

    std::unique_ptr<ParallelScheduler> build() const;

  private:
    BatchSchedulerJob job;

    Optimizer *optimizer;

    size_t max_thread = 1;
    bool multiple_thread_per_device = false;
    std::vector<cl::Device> devices = utils::cl_wrapper.getDevices();
  };

}   // namespace nnet