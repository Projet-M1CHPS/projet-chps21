#pragma once
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS 1
#include "clKernelMap.hpp"
#include "clQueueHandler.hpp"
#include <CL/opencl.hpp>
#include <iostream>
#include <map>
#include <shared_mutex>
#include <thread>
#include <tscl.hpp>

namespace utils {
  class clWrapper {
  public:
    clWrapper(const clWrapper &other);
    clWrapper &operator=(const clWrapper &other);

    clWrapper &operator=(clWrapper &&other);
    clWrapper(clWrapper &&other);

    explicit clWrapper(cl::Platform &platform,
                       const std::filesystem::path &kernels_search_path");

    static std::unique_ptr<clWrapper> makeDefaultWrapper(std::filesystem::path kernels_search_path = "kernels");

    cl::Platform getPlatform() {
      return platform; }
    cl::Context getContext() {
      return context; }
    cl::CommandQueue getDefaultQueue() {
      return default_queue; }

    /**
     * @brief Returns the default queue handler containing all available devices in this wrapper.
     * @return The default queue handler that may already be in used
     */
    clQueueHandler &getDefaultQueueHandler() {
      return default_queue_handler; }

    /**
     * @brief Returns a new queue handler containing all available devices in this wrapper.
     * @return a new queue handler
     */
    clQueueHandler makeQueueHandler(cl::QueueProperties properties = {});
    cl::Device getDefaultDevice() {
      return default_device; }

    clKernelMap &getKernels() {
      return *kernels; }

  private:
    std::shared_mutex main_mutex;

    cl::Platform platform;
    cl::Context context;

    cl::Device default_device;
    std::vector<cl::Device> devices;

    cl::CommandQueue default_queue;
    clQueueHandler default_queue_handler;

    std::shared_ptr<clKernelMap> kernels;
  };
}   // namespace utils
