#pragma once
#include "BatchLocation.hpp"
#include "BatchOptimizationScheduler.hpp"

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

  /**
   * @brief A dispatcher that can dispatch batches to multiple computing devices.
   */
  class ParallelScheduler::Dispatcher {
  public:
    virtual ~Dispatcher() = default;
    virtual void dispatch(BatchLocation &starting_location, size_t work_size,
                          Optimizer::Operation &op) = 0;
  };

  /**
   * @brief Describes the resources allocated to the parallel scheduler.
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

  /**
   * @brief Concrete builder for ParallelScheduler
   */
  class ParallelScheduler::Builder {
  public:
    /**
     * @brief Define the job that needs to be scheduled
     * @param njob
     */
    void setJob(const BatchSchedulerJob &njob) { job = njob; }

    /**
     * @brief Define the allocated threads for the parallel scheduler
     * @param nmax_thread
     * @param nallow_multiple_thread_per_device
     */
    void setMaxThread(size_t nmax_thread, bool nallow_multiple_thread_per_device) {
      max_thread = nmax_thread;
      multiple_thread_per_device = nallow_multiple_thread_per_device;
    }

    /**
     * @brief Defines the allowed devices for the parallel scheduler. If not set, all devices in the
     * clWrapper are used.
     * @param ndevices
     */
    void setDevices(const std::vector<cl::Device> &ndevices) { devices = ndevices; }

    /**
     * @brief Defines the optimizer that the scheduler will use
     * @param new_optimizer
     */
    void setOptimizer(Optimizer &new_optimizer) { this->optimizer = &new_optimizer; }

    /**
     * @brielf Builds the parallel scheduler
     * @return
     */
    std::unique_ptr<ParallelScheduler> build() const;

  private:
    BatchSchedulerJob job;

    Optimizer *optimizer;

    size_t max_thread = 1;
    bool multiple_thread_per_device = false;
    std::vector<cl::Device> devices = utils::cl_wrapper.getDevices();
  };

}   // namespace nnet