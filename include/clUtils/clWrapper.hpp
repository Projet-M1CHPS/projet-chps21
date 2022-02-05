#pragma once
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

    cl::Platform getPlatform() { return platform; }
    cl::Context getContext() { return context; }
    cl::CommandQueue getDefaultQueue() { return default_queue; }
    cl::Device getDefaultDevice() { return default_device; }

    cl::Program getProgram(const std::string &program_name);

  private:
    clWrapper() = default;

    std::shared_mutex main_mutex;
    cl::Platform platform;
    cl::Context context;
    cl::CommandQueue default_queue;
    cl::Device default_device;
    std::map<std::string, cl::Program> programs;
  };

  void printAvailablePlatforms();
}   // namespace utils
