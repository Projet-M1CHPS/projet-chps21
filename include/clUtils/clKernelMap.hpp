#pragma once
#include "clKernelMap.hpp"
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS 1
#include <CL/cl.hpp>
#include <filesystem>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_map>

namespace utils {
  class clKernelMap {
  public:
    explicit clKernelMap(cl::Context &context, std::filesystem::path kernels_path = "kernels");

    cl::Program &getProgram(const std::string &program_name);
    cl::Kernel getKernel(const std::string &program_name, const std::string &kernel_name);

  private:
    std::shared_mutex map_mutex;
    cl::Context context;

    std::unordered_map<std::string, cl::Program> map;
    std::filesystem::path search_path;
  };
}   // namespace utils
