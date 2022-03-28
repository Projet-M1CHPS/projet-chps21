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
    clWrapper(const clWrapper &other) noexcept { *this = other; }
    clWrapper &operator=(const clWrapper &other) noexcept;

    clWrapper(clWrapper &&other) noexcept { *this = std::move(other); }
    clWrapper &operator=(clWrapper &&other) noexcept;

    explicit clWrapper(cl::Platform &platform, size_t device_id,
                       const std::filesystem::path &kernels_search_path = "kernels") noexcept;

    explicit clWrapper(cl::Platform &platform,
                       const std::filesystem::path &kernels_search_path = "kernels") noexcept
        : clWrapper(platform, 0, kernels_search_path) {}

    static std::unique_ptr<clWrapper>
    makeDefault(const std::filesystem::path &kernels_search_path = "kernels") noexcept;

    static clWrapper &setDefault(clWrapper &wrapper) noexcept;

    cl::Platform getPlatform() { return platform; }
    cl::Context getContext() { return context; }
    cl::CommandQueue &getDefaultQueue() { return default_queue; }

    /**
     * @brief Returns the default queue handler containing all available devices in this wrapper.
     * @return The default queue handler that may already be in used
     */
    clQueueHandler &getDefaultQueueHandler() { return default_queue_handler; }

    /**
     * @brief Returns a new queue handler containing all available devices in this wrapper.
     * @return a new queue handler
     */
    clQueueHandler makeQueueHandler(cl::QueueProperties properties = {});
    cl::Device getDefaultDevice() { return default_device; }

    clKernelMap &getKernels() { return *kernels; }

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

  extern clWrapper cl_wrapper;
}   // namespace utils
