#include "clQueueHandler.hpp"

namespace utils {


  clQueueHandler::clQueueHandler(const cl::Context &context, const std::vector<cl::Device> &devices,
                                 cl::QueueProperties properties) {
    // Create a queue for every devices
    for (auto &device : devices) { queues.emplace_back(context, device, properties); }
  }

  clQueueHandler &clQueueHandler::operator=(const clQueueHandler &other) {
    // No need to lock the mutex for copying
    for (auto &q : other.queues) { queues.push_back(q); }
    current_queue_index = 0;
    return *this;
  }

  clQueueHandler &clQueueHandler::operator=(clQueueHandler &&other) noexcept {
    // No need to lock the mutex for copying
    for (auto &q : other.queues) { queues.push_back(q); }

    // Since this is a move copy, we need to clear the other queue
    std::scoped_lock lock(other.queue_mutex);
    other.queues.clear();
    return *this;
  }

  void clQueueHandler::enqueue(cl::Kernel &kernel, const cl::NDRange &offset,
                               const cl::NDRange &global, const cl::NDRange &local,
                               const std::vector<cl::Event> *events_queue, cl::Event *event) {
    // We cannot fail silently as the user may except the kernels to be executed
    // Which could lead to hardly debuggable errors
    if (queues.empty()) { throw std::runtime_error("No queues available for kernel dispatch"); }

    // Needed if multiple threads are enqueuing kernels
    std::scoped_lock<std::mutex> lock(queue_mutex);

    // Enqueue the kernel on the current queue and increment the queue index, looping if necessary
    auto &queue = queues[current_queue_index];
    current_queue_index = (current_queue_index + 1) % queues.size();
    queue.enqueueNDRangeKernel(kernel, offset, global, local, events_queue, event);
  }

  cl::CommandQueue clQueueHandler::getCurrentQueue() {
    // Needed if multiple threads are enqueuing kernels
    std::scoped_lock<std::mutex> lock(queue_mutex);
    return queues[current_queue_index];
  }

  void clQueueHandler::waitAll() {
    for (auto &queue : queues) { queue.finish(); }
  }
}   // namespace utils