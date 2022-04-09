#pragma once
#include "Optimizer.hpp"
#include "math/clFTensor.hpp"
#include <iostream>
#include <utility>
#include <vector>

namespace nnet {

  class OptimizerBatchPolicy {
  public:
    OptimizerBatchPolicy();
    OptimizerBatchPolicy(size_t max_concurrent_thread, size_t max_thread_per_device,
                         const std::vector<cl::Device> &allowed_devices);

    std::vector<cl::Device> getAllowedDevices() const;
    size_t getMaxConcurrentThread() const;
    size_t getMaxThreadPerDevice() const;

  private:
    size_t batch_size;
    size_t max_concurrent_thread;
    size_t max_thread_per_device;
    std::vector<cl::Device> allowed_devices;
  };

  class OptimizerBatchInfo {
  public:
    OptimizerBatchInfo();

    std::chrono::microseconds getTotalTime();

  private:
    OptimizerBatchPolicy *policy;
    std::chrono::microseconds mean_batch_duration;
    std::chrono::microseconds mean_batch_duration_per_device;
  };

  class OptimizerBatchScheduler {
  public:
    OptimizerBatchScheduler(size_t batch_size, Optimizer &optimizer, OptimizerBatchPolicy &policy)
        : batch_size(batch_size), optimizer(&optimizer), batch_policy(&policy) {}

    static OptimizerBatchScheduler makeDefault(size_t batch_size, Optimizer &optimizer);

    OptimizerBatchInfo run(const std::vector<math::clFTensor> &inputs,
                           const std::vector<math::clFTensor> &targets) {
      auto batch_operation = optimizer->makeBatchOperation();

      auto device = cl::Device::getDefault();
      for (size_t i = 0; i < inputs.size(); i++) {
        (*batch_operation)(inputs[i], targets[i], device);
        batch_operation->updateModel();
      }
      optimizer->update();
    }

  protected:
    OptimizerBatchPolicy *batch_policy;
    Optimizer *optimizer;
    size_t batch_size;
  };

}   // namespace nnet
