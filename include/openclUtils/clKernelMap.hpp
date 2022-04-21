#pragma once
#include "clKernelMap.hpp"
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS 1
#include <CL/opencl.hpp>
#include <filesystem>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_map>

namespace utils {
  /*
   * @brief Class for managing OpenCL kernels
   */
  class clKernelMap {
  public:
    /**
     * @brief Build a map with the given kernel path and context. Kernels are lazy loaded and
     * compiled for every devices in the context
     * @param context
     * @param kernels_path
     */
    explicit clKernelMap(cl::Context &context, std::filesystem::path kernels_path = "kernels");

    /**
     * @brief Fetch a program from the map, lazy loading it if not present
     * @param program_name The nae of the program to fetch
     * @return A reference to the loaded program, throws on error
     */
    cl::Program &getProgram(const std::string &program_name);

    /**
     * @brief Retrieve a kernel from a program
     * @param program_name The name of the program, lazy loading it if not present
     * @param kernel_name The name of the kernel to extract from the program
     * @return A new kernel object for the kernel
     */
    cl::Kernel getKernel(const std::string &program_name, const std::string &kernel_name);

  private:
    std::shared_mutex map_mutex;
    cl::Context context;

    std::unordered_map<std::string, cl::Program> map;
    std::filesystem::path search_path;
  };
}   // namespace utils
