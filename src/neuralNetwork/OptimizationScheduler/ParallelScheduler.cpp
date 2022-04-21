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
      DeviceResource(size_t n_thread, cl::Device &device) : n_thread(n_thread) {
        for (size_t i = 0; i < n_thread; ++i) {
          queues.emplace_back(utils::cl_wrapper.getContext(), device);
        }
      }

      size_t getThreadCount() const { return n_thread; }
      cl::CommandQueue &getQueue(size_t index) { return queues[index]; }

    private:
      size_t n_thread;
      std::vector<cl::CommandQueue> queues;
    };

    class DefaultDispatcher final : public ParallelScheduler::Dispatcher {
    public:
      explicit DefaultDispatcher(const ParallelScheduler::Policy &policy);
      void dispatch(BatchLocation &progression, size_t batch_size,
                    Optimizer::Operation &op) override;

    private:
      static void runBatch(size_t thread_rank, BatchLocation progression, size_t count,
                           cl::CommandQueue queue, Optimizer::Operation &op);

      std::unique_ptr<asio::thread_pool> worker_pool;
      std::vector<DeviceResource> resources;
      size_t thread_pool_size;
    };

    DefaultDispatcher::DefaultDispatcher(const ParallelScheduler::Policy &policy) {
      size_t total_thread = policy.getMaxThread();
      if (total_thread == 0) total_thread = std::thread::hardware_concurrency();


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
      worker_pool = std::make_unique<asio::thread_pool>(thread_pool_size);
    }

    void DefaultDispatcher::dispatch(BatchLocation &progression, size_t batch_size,
                                     Optimizer::Operation &op) {
      size_t local_work_size = batch_size / thread_pool_size;
      size_t remainder = batch_size % thread_pool_size;
      std::list<std::future<void>> futures;

      // Ensure we have enough caches for all the threads
      op.reserveCaches(thread_pool_size);
      size_t thread_rank = 0;

      // TODO: Refactor me!
      for (auto &resource : resources) {
        for (size_t j = 0; j < resource.getThreadCount(); j++) {
          size_t thread_work = local_work_size;

          if (remainder) {
            thread_work++;
            remainder--;
          }

          // If there isn't enough data to fill the thread pool, break
          if (thread_work == 0) break;
          auto thread_lambda = [=, &c = resource.getQueue(j), &op] {
            return runBatch(thread_rank, progression, thread_work, c, op);
          };
          futures.emplace_back(
                  asio::post(*this->worker_pool, std::packaged_task<void(void)>(thread_lambda)));
          progression.progress(local_work_size);
          thread_rank++;
        }
      }

      // Wait for all threads to finish
      // Note that using thread_pool join effectively kills the thread pool
      for (auto &f : futures) { f.get(); }
    }

    void DefaultDispatcher::runBatch(size_t thread_rank, BatchLocation progression, size_t count,
                                     cl::CommandQueue queue, Optimizer::Operation &op) {
      for (size_t i = 0; i < count;) {
        size_t work_size = std::min(count - i, progression.getBatchRemainder());

        clFTensor current_input = progression.getInputSlice(work_size);
        clFTensor current_target = progression.getTargetSlice(work_size);

        op(thread_rank, current_input, current_target, queue);
        progression.progress(work_size);
        i += work_size;
      }
      queue.finish();
    }
  }   // namespace


  ParallelScheduler::ParallelScheduler(const BatchSchedulerJob &job, Optimizer &optimizer,
                                       std::unique_ptr<Dispatcher> dispatcher)
      : BatchOptimizationScheduler(job), batch_dispatcher(std::move(dispatcher)),
        optimizer(&optimizer) {
    bool tensors_ok = checkTensorsSize(job.getInputs(), job.getTargets());
    if (not tensors_ok or not job.isValid()) {
      throw std::runtime_error("ParallelScheduler::ParallelScheduler: Tensors size mismatch");
    }
    optimizer_operation = optimizer.makeOperation();
  }

  ParallelScheduler ParallelScheduler::makeWithDefaultDispatcher(const BatchSchedulerJob &job,
                                                                 Optimizer &optimizer,
                                                                 const Policy &policy) {
    return {job, optimizer, std::make_unique<DefaultDispatcher>(policy)};
  }

  void ParallelScheduler::run() {
    // TODO: Refactor me!
    epochStart();
    size_t global_work_size = getJob().getGlobalWorkSize();
    size_t batch_size = getJob().getBatchSize();

    BatchLocation progression(getJob().getInputs(), getJob().getTargets());

    for (size_t current_size = 0; current_size < global_work_size; current_size += batch_size) {
      size_t current_batch_size = std::min(global_work_size - current_size, batch_size);
      batch_dispatcher->dispatch(progression, current_batch_size, *optimizer_operation);
      updateModel();
    }

    optimizer->update();
    endEpoch();
  }

  void ParallelScheduler::updateModel() {
    optimizer_operation->updateModel(utils::cl_wrapper.getDefaultQueue());
  }

  void ParallelScheduler::print(std::ostream &os) const {
    os << "ParallelScheduler: " << std::endl;
    os << "\tBatch size: " << getJob().getBatchSize() << std::endl;
    os << "\tGlobal work size: " << getJob().getGlobalWorkSize() << std::endl;
  }

  void ParallelScheduler::epochStart() {}
  void ParallelScheduler::endEpoch() {}

  ParallelScheduler::Policy::Policy(size_t max_thread, bool multiple_thread_per_device,
                                    std::vector<cl::Device> devices)
      : max_thread(max_thread), multiple_thread_per_device(multiple_thread_per_device),
        devices(std::move(devices)) {}

  std::unique_ptr<ParallelScheduler> ParallelScheduler::Builder::build() const {
    if (not optimizer or devices.empty() or not job.isValid()) {
      throw std::runtime_error("ParallelScheduler::Builder: not all required parameters are set");
    }

    Policy policy(max_thread, multiple_thread_per_device, devices);
    auto scheduler = ParallelScheduler::makeWithDefaultDispatcher(job, *optimizer, policy);
    return std::make_unique<ParallelScheduler>(std::move(scheduler));
  }

}   // namespace nnet