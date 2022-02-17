#pragma once
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS 1
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
    clWrapper(clWrapper &other);

    static clWrapper makeDefaultWrapper();
    static clWrapper fromPlatform(const cl::Platform &platform);

    cl::Platform getPlatform() { return platform; }
    cl::Context getContext() { return context; }
    cl::CommandQueue getDefaultQueue() { return default_queue; }
    /**
     * @brief Returns the default queue handler containing all available devices in this wrapper.
     * @return The default queue handler that may already be in used
     */
    clQueueHandler &getDefaultQueueHandler() { return default_queue_handler; }

    /**
     * @brief Returns a new queue handler containing all available devices in this wrapper.
     * @return a new queue handler
     */
    clQueueHandler &makeQueueHandler();
    cl::Device getDefaultDevice() { return default_device; }

    cl::Program getProgram(const std::string &program_name);
    cl::Kernel getKernel(const std::string &program_name, const std::string &kernel_name);

  private:
    std::shared_mutex main_mutex;
    cl::Platform platform;
    cl::Context context;
    cl::CommandQueue default_queue;
    clQueueHandler default_queue_handler;
    cl::Device default_device;
    std::map<std::string, cl::Program> programs;
  };

  void printAvailablePlatforms();
}   // namespace utils
