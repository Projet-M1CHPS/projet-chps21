#pragma once
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS 1
#include "clKernelMap.hpp"
#include <CL/opencl.hpp>
#include <boost/dll.hpp>
#include <iostream>
#include <map>
#include <shared_mutex>
#include <thread>
#include <tscl.hpp>

namespace utils {

  class clWrapper {
  public:
    /**
     * @brief Initializes the openCL environment before use, setting up default platform/device, and
     * setting up the global wrapper This function must explicitly be called once (and only once)
     * before any call to OpenCL functions to ensure coherency.  This function is thread-safe.
     *
     * If this function is called twice, the program will terminate with an error, and no exception
     * shall be raised.
     *
     * @param wrapper
     * @return
     */
    static clWrapper &initOpenCL(clWrapper &wrapper) noexcept;

    clWrapper() noexcept = default;

    clWrapper(const clWrapper &other) { *this = other; }
    clWrapper &operator=(const clWrapper &other);

    clWrapper(clWrapper &&other) noexcept { *this = std::move(other); }
    clWrapper &operator=(clWrapper &&other) noexcept;

    explicit clWrapper(cl::Platform &platform, size_t device_id,
                       const std::filesystem::path &kernels_search_path = "kernels");

    explicit clWrapper(cl::Platform &platform,
                       const std::filesystem::path &kernels_search_path = "kernels")
        : clWrapper(platform, 0, kernels_search_path) {}

    /**
     * @brief Create a default wrapper using the first available platform, prioritizing GPU devices.
     * @param kernels_search_path
     * @return
     */
    static std::unique_ptr<clWrapper>
    makeDefault(const std::filesystem::path &kernels_search_path = "kernels");

    cl::Platform getPlatform() { return platform; }
    cl::Context getContext() { return context; }
    cl::CommandQueue &getDefaultQueue() { return default_queue; }

    std::vector<cl::Device> &getDevices() { return devices; }
    const std::vector<cl::Device> &getDevices() const { return devices; }

    cl::Device getDefaultDevice() { return default_device; }

    clKernelMap &getKernels() { return *kernels; }

  private:
    std::shared_mutex main_mutex;

    cl::Platform platform;
    cl::Context context;

    cl::Device default_device;
    std::vector<cl::Device> devices;

    cl::CommandQueue default_queue;

    std::shared_ptr<clKernelMap> kernels;
  };

  extern clWrapper cl_wrapper;
}   // namespace utils
