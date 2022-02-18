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

  void clQueueHandler::waitAll() {
    for (auto &queue : queues) { queue.finish(); }
  }
}   // namespace utils