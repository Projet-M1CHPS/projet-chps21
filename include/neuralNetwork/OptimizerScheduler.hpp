#pragma once
#include "Optimizer.hpp"
#include "math/clFTensor.hpp"
#include <boost/asio/thread_pool.hpp>
#include <iostream>
#include <utility>
#include <vector>

namespace nnet {

  /**
   * @brief Policy describing the resources allocated for the scheduler, including which device to
   * use, and the number of threads to use.
   */
  class OptimizerSchedulerPolicy {
  public:
    friend std::ostream &operator<<(std::ostream &os, const OptimizerSchedulerPolicy &policy);

    /**
     * @brief Return a default policy, allowing all devices to be used with the hardware
     * concurrency, a single thread per device.
     * @return A generic policy
     */
    static OptimizerSchedulerPolicy defaultPolicy() {
      return {2, true};
    }


    /**
     * @brief Creates a new policy
     * @param max_concurrent_thread The maximum number of threads to use concurrently
     * @param multiple_thread_per_device If true, allows the scheduler to use multiple threads per
     * device By default, the scheduler will use a single thread per device. If this is set, any
     * extra threads will be dispatched among the devices.
     * @param allowed_devices A list of the devices that the scheduler is allowed to use. If empty,
     * uses all devices available in the default wrapper
     */
    OptimizerSchedulerPolicy(size_t max_concurrent_thread, bool multiple_thread_per_device,
                             const std::vector<cl::Device> &allowed_devices = {});

    /**
     * @brief Return a list of the devices that the scheduler is allowed to use
     * @return
     */
    const std::vector<cl::Device> &getAllowedDevices() const { return allowed_devices; }

    /**
     * @brief If true, allows the scheduler to dispatch extra threads among the devices
     * @return True if the scheduler may dispatch extra threads, false otherwise.
     */
    bool hasMultipleThreadPerDevice() const { return multiple_thread_per_device; }

    /**
     * @brief Return the maximum number of threads to use concurrently
     * @return
     */
    size_t getMaxConcurrentThread() const { return max_concurrent_thread; }

  private:
    size_t max_concurrent_thread;
    bool multiple_thread_per_device;
    std::vector<cl::Device> allowed_devices;
  };

  /**
   * @brief Stores metadata about the last scheduler run, including multiple timers.
   *
   * Note: This is a pretty rough implementation, but will do the job for the time being
   */
  class OptimizerSchedulerInfo {
  public:
    friend std::ostream &operator<<(std::ostream &os, const OptimizerSchedulerInfo &info);

    OptimizerSchedulerInfo(const OptimizerSchedulerPolicy &policy);

    const OptimizerSchedulerPolicy& getPolicy() const { return *policy; }

    std::chrono::milliseconds getTotalTime() const { return std::chrono::duration_cast<std::chrono::milliseconds>(total_time); }

    void setTotalTime(std::chrono::microseconds time) { total_time = time; }

    std::chrono::microseconds getBatchTime() const { return time_per_batch; }

    void setBatchTime(std::chrono::microseconds time) { time_per_batch = time; }

    float getBatchPerSecond() const {
      return 1.0f / std::chrono::duration_cast<std::chrono::seconds>(time_per_batch).count();
    }

    std::chrono::microseconds getInputTime() const { return time_per_input; }

    void setInputTime(std::chrono::microseconds time) { time_per_input = time; }

    float getInputPerSecond() const {
      return 1.0f / std::chrono::duration_cast<std::chrono::seconds>(time_per_input).count();
    }

    std::chrono::microseconds getModelUpdateTime(std::chrono::microseconds time) const {
      return model_update_duration;
    }

    void setModelUpdateTime(std::chrono::microseconds time) { model_update_duration = time; }

  private:
    const OptimizerSchedulerPolicy *policy;

    std::chrono::microseconds total_time;
    std::chrono::microseconds time_per_batch;
    std::chrono::microseconds time_per_input;
    std::chrono::microseconds model_update_duration;
  };

  class OptimizerScheduler {
  public:
    friend std::ostream &operator<<(std::ostream &os, const OptimizerScheduler &scheduler);

    OptimizerScheduler(size_t batch_size, Optimizer &optimizer,
                       const OptimizerSchedulerPolicy &policy);

    OptimizerSchedulerInfo run(const std::vector<math::clFTensor> &inputs,
                               const std::vector<math::clFTensor> &targets);

  protected:

    // We need to allocate the worker_pool on the heap, since it is not copyable and we don't know
    // the number of thread in advance
    std::unique_ptr<boost::asio::thread_pool> worker_pool;
    size_t worker_pool_size;

    Optimizer *optimizer;
    const OptimizerSchedulerPolicy *policy;
    const size_t batch_size;
  };

}   // namespace nnet
