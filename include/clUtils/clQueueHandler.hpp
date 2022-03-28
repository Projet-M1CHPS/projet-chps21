#pragma once
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS 1
#include <CL/opencl.hpp>
#include <mutex>

namespace utils {

  /**
   * @brief Queue handler for kernel dispatching
   * Wraps a set of queue objects and provides a simple interface for dispatching kernels to them.
   * Since this is a simple implementation, kernels are dispatched in round-robin.
   *
   * This object is copyable, which can be used for parallel enqueues.
   */
  class clQueueHandler {
  public:
    /**
     * @brief Builds an empty queue handler. Enqueuing kernels will throw an error on empty queue.
     *
     *
     * We allow building empty queues for error handling and easier initialization of other objects
     * @param queues The set of queues to use
     */
    clQueueHandler() = default;

    /**
     * @brief Copies the queue of the other handler
     * The new object will not share the same mutex, but will use the same queue.
     * This can be used to for parallel kernel enqueues.
     * @param other
     */
    clQueueHandler(const clQueueHandler &other) { *this = other; }
    clQueueHandler &operator=(const clQueueHandler &other);

    /**
     * @brief Locks the other handler and take ownership of its queues
     * Be warned that the other handler is left in an invalid state and should not be used anymore.
     * Take caution when using this function in code where the handler is used for parallel
     * enqueuing
     * @param other
     */
    clQueueHandler(clQueueHandler &&other) noexcept { *this = other; }
    clQueueHandler &operator=(clQueueHandler &&other) noexcept;

    /**
     * @brief Builds a queue for each device using the given flags
     * @param devices The devices to build queues for
     * @param properties The queues properties
     */
    clQueueHandler(const cl::Context &context, const std::vector<cl::Device> &devices,
                   cl::QueueProperties properties = {});
    /**
     * @brief Returns the next queue that will be used for enqueuing kernels
     * @return
     */
    cl::CommandQueue &current() {
      std::scoped_lock<std::mutex> lock(queue_mutex);
      return queues[current_queue_index];
    }

    /**
     * @brief Returns a queue and advances the queue index
     * @return The next queue
     */
    cl::CommandQueue &next() {
      std::scoped_lock<std::mutex> lock(queue_mutex);
      auto &res = queues[current_queue_index];
      current_queue_index = (current_queue_index + 1) % queues.size();
      return res;
    }

    /**
     * @brief returns a raw pointer to the queues array
     * This is useful when using clblas that takes a list of queues for kernel dispatching
     * @return A pointer to the beginning of the queue array
     */
    cl::CommandQueue *getQueues() { return queues.data(); }

    /**
     * @return The numbers of queues in this handler
     */
    size_t size() const { return queues.size(); }

    /**
     * @return True if this handler has no queues
     */
    bool empty() const { return queues.empty(); }

    /**
     * @brief Waits for all queues to finish their enqueued kernels
     */
    void waitAll();

  private:
    std::mutex queue_mutex;
    size_t current_queue_index = 0;
    std::vector<cl::CommandQueue> queues;
  };
}   // namespace utils