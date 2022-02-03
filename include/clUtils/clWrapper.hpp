#pragma once
#include <CL/opencl.hpp>
#include <iostream>
#include <map>
#include <tscl.hpp>

namespace utils {
  class clWrapper {
  public:
    cl::Platform getPlatform();
    cl::Context getContext();
    cl::CommandQueue getDefaultQueue();
    cl::Device getDefaultDevice();

    cl::Program getProgram(const std::string &program_name);

  private:
    cl::Platform platform;
    cl::Context context;
    cl::CommandQueue default_queue;
    cl::Device default_device;
    std::map<std::string, cl::Program> programs;
  };
}   // namespace utils
