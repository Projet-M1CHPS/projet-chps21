#include "OptimizerScheduler.hpp"

#include <boost/asio/io_service.hpp>

using namespace std::chrono_literals;
using namespace boost::asio;

namespace nnet {

  std::ostream &operator<<(std::ostream &os, const OptimizerSchedulerPolicy &policy) {
    os << "OptimizerSchedulerPolicy: " << std::endl;
    os << "\tmax_concurrent_thread: " << policy.max_concurrent_thread << std::endl;
    os << "\tmultiple thread per device: " << (policy.multiple_thread_per_device ? "true" : "false")
       << std::endl;
    os << "\tAllowed devices: ";
    for (auto &device : policy.allowed_devices) {
      os << "\t\t" << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    }
  }

  OptimizerSchedulerPolicy::OptimizerSchedulerPolicy(size_t max_concurrent_thread,
                                                     bool multiple_thread_per_device,
                                                     const std::vector<cl::Device> &allowed_devices)
      : max_concurrent_thread(max_concurrent_thread),
        multiple_thread_per_device(multiple_thread_per_device), allowed_devices(allowed_devices) {
    if (not allowed_devices.empty()) return;
    // If no device is specified, use all devices
    // available in the default wrapper
    this->allowed_devices = utils::cl_wrapper.getDevices();
  }

  std::ostream &operator<<(std::ostream &os, const OptimizerSchedulerInfo &info) {
    os << "OptimizerSchedulerInfo: ";
    os << "\tTotal time: " << info.getTotalTime().count() << "us" << std::endl;
    os << "\tMean batch duration: " << info.getBatchTime().count() << "us" << std::endl;
    os << "\tMean batch/s: " << info.getBatchPerSecond() << "us" << std::endl;
    os << "\tMean duration per input: " << info.getInputTime().count() << "us" << std::endl;
    os << "\tMean input/s: " << info.getInputPerSecond() << "us" << std::endl;
    os << "\tMean model update duration: " << info.getInputPerSecond() << "us" << std::endl;
    return os;
  }

  std::ostream &operator<<(std::ostream &os, const OptimizerScheduler &scheduler) {
    os << "OptimizerScheduler: " << std::endl;
    os << "Batch size: " << scheduler.batch_size;
    os << *scheduler.policy << std::endl;
    return os;
  }

  OptimizerSchedulerInfo::OptimizerSchedulerInfo(OptimizerSchedulerPolicy &policy)
      : policy(&policy), total_time(0s), time_per_input(0s), time_per_batch(0s),
        model_update_duration(0s) {}

  OptimizerScheduler::OptimizerScheduler(size_t batch_size, Optimizer &optimizer,
                                         const OptimizerSchedulerPolicy &policy)
      : batch_size(batch_size), optimizer(&optimizer), policy(&policy) {
    size_t nthread = policy.getMaxConcurrentThread();
    // If the policy does not allow multiple thread per device, restrict the number of thread to the
    // number of device
    if (not policy.hasMultipleThreadPerDevice()) nthread = policy.getAllowedDevices().size();
    worker_pool = std::make_unique<thread_pool>(nthread);
  }

  std::pair<size_t, size_t> dispatchBatch(size_t batch_size, std::pair<size_t, size_t> indexes,
                                          const std::vector<math::clFTensor> &inputs,
                                          const std::vector<math::clFTensor> &targets,
                                          OptimizerOperation &batch_operation) {}

  void threadRunBatch(const std::vector<math::clFTensor> *inputs,
                      const std::vector<math::clFTensor> *targets, OptimizerOperation *operation,
                      size_t start_index, const cl::Device &device) {}


  OptimizerSchedulerInfo OptimizerScheduler::run(const std::vector<math::clFTensor> &inputs,
                                                 const std::vector<math::clFTensor> &targets) {
    if (inputs.size() != targets.size()) {
      throw std::runtime_error("OptimizerScheduler: inputs and targets size mismatch");
    }

    auto optimizer_operation = optimizer->makeBatchOperation();

    auto device = cl::Device::getDefault();
    io_service io_service;
    io_service::work work(io_service);

    size_t global_work_size = 0, tensor_index = 0, local_index = 0;
    for (auto &t : inputs) { global_work_size += t.getDepth(); }

    for (size_t current_size = 0; current_size < global_work_size; current_size += batch_size) {
      size_t current_batch_size = std::min(global_work_size - current_size, batch_size);
      std::tie(tensor_index, local_index) =
              dispatchBatch(current_batch_size, {tensor_index, local_index}, inputs, targets,
                            *optimizer_operation);
      optimizer_operation->updateModel();
    }
    optimizer->update();
  }
}   // namespace nnet
