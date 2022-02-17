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
   */
  class clQueueHandler {
  public:
    clQueueHandler() = delete;
    clQueueHandler(const clQueueHandler &) = delete;

    clQueueHandler(clQueueHandler &&other);
    clQueueHandler &operator=(clQueueHandler &&other);

    /**
     * @brief Builds a queue for each device using the given flags
     * @param devices The devices to build queues for
     * @param properties The queues properties
     */
    clQueueHandler(const std::vector<cl::Device> &devices, cl_command_queue_properties properties);

    /**
     * @brief Dispatch a kernel to the next queue
     * @param kernel The kernel to dispatch, with arguments already set
     * @param events_queue The queue to use for the event
     * @param event The event associated with this kernel
     */
    void enqueue(cl::Kernel &kernel, const std::vector<cl::Event> *events_queue = nullptr,
                 cl::Event *event = nullptr);

    cl::CommandQueue getCurrentQueue() const;

    /**
     * @return The numbers of queues in this handler
     */
    size_t size() const;

    /**
     * @return True if this handler has no queues
     */
    bool empty() const;

    /**
     * @brief Waits for all queues to end
     */
    void waitAll();

  private:
    std::mutex mutex;
    int current_queue_index;
    std::vector<cl::CommandQueue> queues;
  };
}   // namespace utils