#include "OptimizationScheduler.hpp"

#include <boost/asio.hpp>
#include <boost/bind/bind.hpp>
#include <future>

using namespace std::chrono_literals;
using namespace boost;
using namespace math;

namespace nnet {

  namespace {

    std::pair<size_t, size_t> incrementIndexes(const std::vector<clFTensor> &inputs,
                                               size_t tensor_index, size_t local_index) {
      while (local_index > 0 and local_index >= inputs[tensor_index].getDepth()) {
        local_index -= inputs[tensor_index].getDepth();
        tensor_index++;
        // If we have reached the end of the input sets, loop back to the beginning
        if (tensor_index >= inputs.size()) tensor_index = 0;
      }
      return {tensor_index, local_index};
    }

    class BatchDispatcher {
    public:
      BatchDispatcher(const std::vector<clFTensor> &inputs, const std::vector<clFTensor> &targets,
                      OptimizerOperation &op, asio::thread_pool &worker_pool, size_t pool_size);

      void dispatch(size_t batch_size);

    private:
      void runBatch(size_t start_tensor, size_t start_index, size_t count);

      const std::vector<clFTensor> *inputs, *targets;
      OptimizerOperation *op;
      asio::thread_pool *worker_pool;
      size_t thread_pool_size;
      size_t tensor_index = 0, local_index = 0;
    };

    BatchDispatcher::BatchDispatcher(const std::vector<clFTensor> &inputs,
                                     const std::vector<clFTensor> &targets, OptimizerOperation &op,
                                     asio::thread_pool &worker_pool, size_t pool_size)
        : inputs(&inputs), targets(&targets), op(&op), worker_pool(&worker_pool),
          thread_pool_size(pool_size) {}

    void BatchDispatcher::dispatch(size_t batch_size) {
      size_t local_work_size = batch_size / thread_pool_size;
      size_t remainder = batch_size % thread_pool_size;
      std::vector<std::future<void>> futures;

      auto lambda = [this](size_t tindex, size_t lindex, size_t count) {
        runBatch(tindex, lindex, count);
      };

      for (size_t i = 0; i < thread_pool_size; ++i) {
        size_t count = local_work_size;
        // if (count == 0) break;
        if (i < remainder) { count++; }
        // TODO: refactor this
        auto task = [lambda, count, ti = tensor_index, li = local_index] {
          return lambda(ti, li, count);
        };
        auto future = asio::post(*worker_pool, std::packaged_task<void()>(task));
        futures.push_back(std::move(future));
        local_index += count;
        std::tie(tensor_index, local_index) = incrementIndexes(*inputs, tensor_index, local_index);
      }
      // Wait for all threads to finish
      // Note that using thread_pool join effectively kills the thread pool
      for (auto &f : futures) { f.get(); }
    }

    void BatchDispatcher::runBatch(size_t start_tensor, size_t start_index, size_t count) {
      cl::Device device = utils::cl_wrapper.getDefaultDevice();

      for (size_t i = 0; i < count;) {
        size_t work_size = std::min(count - i, (*inputs)[start_tensor].getDepth() - start_index);

        clFTensor current_input =
                (*inputs)[start_tensor].slice(start_index, start_index + work_size);
        clFTensor current_target =
                (*targets)[start_tensor].slice(start_index, start_index + work_size);

        (*op)(current_input, current_target, device);
        std::tie(start_tensor, start_index) =
                incrementIndexes(*inputs, start_tensor, start_index + work_size);
        i += work_size;
      }
    }
  }   // namespace

  std::ostream &operator<<(std::ostream &os, const OptimizerSchedulerPolicy &policy) {
    os << "OptimizerSchedulerPolicy: " << std::endl;
    os << "\tmax_concurrent_thread: " << policy.max_concurrent_thread << std::endl;
    os << "\tmultiple thread per device: " << (policy.multiple_thread_per_device ? "true" : "false")
       << std::endl;
    os << "\tAllowed devices: ";
    for (auto &device : policy.allowed_devices) {
      os << "\t\t" << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    }
    return os;
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

  std::ostream &operator<<(std::ostream &os, const OptimizationScheduler &scheduler) {
    os << "OptimizationScheduler: " << std::endl;
    os << "Batch size: " << scheduler.batch_size;
    os << *scheduler.policy << std::endl;
    return os;
  }

  OptimizerSchedulerInfo::OptimizerSchedulerInfo(const OptimizerSchedulerPolicy &policy)
      : policy(&policy), total_time(0s), time_per_batch(0s), time_per_input(0s),
        model_update_duration(0s) {}

  OptimizationScheduler::OptimizerScheduler(size_t batch_size, Optimizer &optimizer,
                                         const OptimizerSchedulerPolicy &policy)
      : optimizer(&optimizer), policy(&policy), batch_size(batch_size) {
    size_t nthread = policy.getMaxConcurrentThread();
    // If the policy does not allow multiple thread per device, restrict the number of thread to the
    // number of device
    if (not policy.hasMultipleThreadPerDevice()) nthread = policy.getAllowedDevices().size();
    worker_pool = std::make_unique<asio::thread_pool>(nthread);
    worker_pool_size = nthread;
  }

  OptimizerSchedulerInfo OptimizationScheduler::run(const std::vector<math::clFTensor> &inputs,
                                                 const std::vector<math::clFTensor> &targets) {
    if (inputs.size() != targets.size()) {
      throw std::runtime_error("OptimizationScheduler: inputs and targets size mismatch");
    }

    auto optimizer_operation = optimizer->makeBatchOperation();

    auto device = cl::Device::getDefault();

    size_t global_work_size = 0;
    for (auto &t : inputs) { global_work_size += t.getDepth(); }

    BatchDispatcher dispatcher(inputs, targets, *optimizer_operation, *worker_pool,
                               worker_pool_size);


    auto start = std::chrono::high_resolution_clock::now();
    for (size_t current_size = 0; current_size < global_work_size; current_size += batch_size) {
      size_t current_batch_size = std::min(global_work_size - current_size, batch_size);
      dispatcher.dispatch(current_batch_size);
      optimizer_operation->updateModel();
    }
    auto end = std::chrono::high_resolution_clock::now();

    optimizer->update();
    OptimizerSchedulerInfo result(*policy);
    result.setTotalTime(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
    return result;
  }
}   // namespace nnet
