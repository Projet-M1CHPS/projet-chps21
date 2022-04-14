#include "ParallelScheduler.hpp"
#include "math/clFTensor.hpp"
#include <boost/asio.hpp>
#include <boost/asio/thread_pool.hpp>
#include <future>

using namespace math;
using namespace boost;

namespace nnet {
  namespace {

    bool checkTensorsSize(const std::vector<clFTensor> &inputs,
                          const std::vector<clFTensor> &targets) {
      if (inputs.size() != targets.size()) return false;

      for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].getDepth() != targets[i].getDepth()) return false;
      }
      return true;
    }

    class DeviceResource {
    public:
      DeviceResource(size_t n_thread, cl::Device device) : n_thread(n_thread), device(device) {}

      size_t getThreadCount() const { return n_thread; }
      cl::Device &getDevice() { return device; }

    private:
      size_t n_thread;
      cl::Device device;
    };

    class DefaultDispatcher final : public ParallelScheduler::Dispatcher {
    public:
      explicit DefaultDispatcher(const ParallelScheduler::Policy &policy);
      void dispatch(BatchProgression &progression, size_t batch_size,
                    Optimizer::Operation &op) override;

    private:
      static void runBatch(BatchProgression progression, size_t count, cl::Device device,
                           Optimizer::Operation &op);

      std::unique_ptr<asio::thread_pool> worker_pool;
      std::vector<DeviceResource> resources;
      size_t thread_pool_size;
    };

    DefaultDispatcher::DefaultDispatcher(const ParallelScheduler::Policy &policy) {
      size_t total_thread = policy.getMaxThread();
      if (total_thread == 0) { total_thread = std::thread::hardware_concurrency(); }
      worker_pool = std::make_unique<asio::thread_pool>(total_thread);


      std::vector<cl::Device> devices =
              policy.getDevices().empty() ? utils::cl_wrapper.getDevices() : policy.getDevices();

      size_t thread_per_device = total_thread / devices.size();
      size_t remainder = total_thread % devices.size();
      if (not policy.hasMultipleThreadPerDevice()) {
        thread_per_device = 1;
        remainder = 0;
      }
      for (size_t i = 0; i < devices.size(); ++i) {
        resources.emplace_back(thread_per_device + (i < remainder ? 1 : 0), devices[i]);
      }
      thread_pool_size = thread_per_device * devices.size() + remainder;
    }

    void DefaultDispatcher::dispatch(BatchProgression &progression, size_t batch_size,
                                     Optimizer::Operation &op) {
      size_t local_work_size = batch_size / thread_pool_size;
      size_t remainder = batch_size % thread_pool_size;
      std::vector<std::future<void>> futures;

      auto lambda = [](BatchProgression p, size_t count, cl::Device d, Optimizer::Operation &op) {
        runBatch(p, count, d, op);
      };

      for (auto &resource : resources) {
        for (size_t j = 0; j < resource.getThreadCount(); j++) {
          size_t thread_work = local_work_size;

          if (remainder) {
            thread_work++;
            remainder--;
          }

          // If there isn't enough data to fill the thread pool, break
          if (thread_work == 0) break;
          auto task = [lambda, thread_work, p = progression, d = resource.getDevice(), &op] {
            return lambda(p, thread_work, d, op);
          };
          auto future = asio::post(*worker_pool, std::packaged_task<void()>(task));
          futures.push_back(std::move(future));
          progression.progress(local_work_size);
        }
      }
      // Wait for all threads to finish
      // Note that using thread_pool join effectively kills the thread pool
      for (auto &f : futures) { f.get(); }

    }

    void DefaultDispatcher::runBatch(BatchProgression progression, size_t count, cl::Device device,
                                     Optimizer::Operation &op) {
      cl::CommandQueue queue(utils::cl_wrapper.getContext(), device);
      for (size_t i = 0; i < count;) {
        size_t work_size = std::min(count - i, progression.getBatchRemainder());

        clFTensor current_input = progression.getInputSlice(work_size);
        clFTensor current_target = progression.getTargetSlice(work_size);

        op(current_input, current_target, queue);
        progression.progress(work_size);
        i += work_size;
      }
    }
  }   // namespace


  ParallelScheduler::ParallelScheduler(const std::vector<math::clFTensor> &inputs,
                                       const std::vector<math::clFTensor> &targets,
                                       size_t batch_size, Optimizer &optimizer,
                                       std::unique_ptr<Dispatcher> dispatcher)
      : BatchOptimizationScheduler(batch_size, optimizer, inputs, targets),
        batch_dispatcher(std::move(dispatcher)) {
    bool tensors_ok = checkTensorsSize(inputs, targets);
    if (not tensors_ok) {
      throw std::runtime_error("ParallelScheduler::ParallelScheduler: Tensors size mismatch");
    }
  }

  ParallelScheduler ParallelScheduler::makeDefaultDispatcher(
          const std::vector<math::clFTensor> &inputs, const std::vector<math::clFTensor> &targets,
          size_t batch_size, Optimizer &optimizer, const Policy &policy) {
    return {inputs, targets, batch_size, optimizer, std::make_unique<DefaultDispatcher>(policy)};
  }

  void ParallelScheduler::run() {
    size_t global_work_size = 0;
    for (auto &t : *input_tensors) { global_work_size += t.getDepth(); }

    BatchProgression progression(*input_tensors, *target_tensors);

    for (size_t current_size = 0; current_size < global_work_size; current_size += batch_size) {
      size_t current_batch_size = std::min(global_work_size - current_size, batch_size);
      batch_dispatcher->dispatch(progression, current_batch_size, *optimizer_operation);
      updateModel();
    }

    optimizer->update();
  }

  void ParallelScheduler::updateModel() { optimizer_operation->updateModel(); }

  void ParallelScheduler::print(std::ostream &os) const {}

  void ParallelScheduler::epochStart() {}
  void ParallelScheduler::endEpoch() {}

  ParallelScheduler::Policy::Policy(size_t max_thread, bool multiple_thread_per_device,
                                    std::vector<cl::Device> devices)
      : max_thread(max_thread), multiple_thread_per_device(multiple_thread_per_device),
        devices(std::move(devices)) {}

}   // namespace nnet