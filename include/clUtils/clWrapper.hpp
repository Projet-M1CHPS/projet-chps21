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
    explicit clWrapper(cl::Platform &platform);

    static clWrapper getDefaultWrapper();
    static void setDefaultWrapper(clWrapper &wrapper);

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
    clQueueHandler makeQueueHandler(cl::QueueProperties properties = {});
    cl::Device getDefaultDevice() { return default_device; }

    /**
     * @brief Lazy load a program and cache it for further query
     * If the program is not found, the app crashes.
     * FIXME: add a default path for programs lookup and improve error handling
     * @param program_name The name of the program to load. This is the path of the file
     * @return The program if it was loader, fatal error otherwise
     */
    cl::Program getProgram(const std::string &program_name);

    /**
     * @brief Lookup the given program in the cache (or lazy load it) and returns the corresponding
     * kernel inside. If the kernel or the program is not found, the app crashes.
     * FIXME: improve error handling
     * @param program_name The name of the program containing the kernel. This is the path of the
     * file
     * @param kernel_name The name of the kernel to retrieve
     * @return The kernel if it was found, fatal error otherwise
     */
    cl::Kernel getKernel(const std::string &program_name, const std::string &kernel_name);

  private:
    std::shared_mutex main_mutex;

    cl::Platform platform;
    cl::Context context;

    cl::Device default_device;
    std::vector<cl::Device> devices;

    cl::CommandQueue default_queue;
    clQueueHandler default_queue_handler;

    std::map<std::string, cl::Program> programs;
  };
}   // namespace utils
