#include "clWrapper.hpp"
#include <fstream>

namespace utils {

  namespace {

    // Placeholder functions, needs more testing on a multi-cpu and multi-gpu architecture
    // The main issue here is letting the user choose which platform to use
    // One option would be to display a menu at runtime and let the user choose, use a configuration
    // file, or combine both
    std::vector<cl::Platform> findCPUPlatform() {
      std::vector<cl::Platform> platforms;
      cl::Platform::get(&platforms);

      std::vector<cl::Platform> cpu_platforms;
      for (auto &platform : platforms) {
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
        if (!devices.empty()) { cpu_platforms.push_back(platform); }
      }
      return platforms;
    }

    std::vector<cl::Platform> findGPUPlatform() {
      std::vector<cl::Platform> platforms;
      cl::Platform::get(&platforms);

      std::vector<cl::Platform> gpu_platforms;
      for (auto &platform : platforms) {
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (!devices.empty()) { gpu_platforms.push_back(platform); }
      }
      return platforms;
    }

    // Loading an OpenCL program requires it to be loaded into a buffer first
    std::string readFile(const std::string &path) {
      std::ifstream t(path);
      if (!t.is_open()) { throw std::runtime_error("Could not open file: " + path); }
      std::string res((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
      return res;
    }
  }   // namespace

  clWrapper::clWrapper(cl::Platform &platform, const std::filesystem::path &kernels_search_path) {
    this->platform = platform;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    default_device = devices[0];
    context = cl::Context(devices);
    default_queue = cl::CommandQueue(context, default_device);
    // By default, we do not enable out-of-order execution for the queue handler
    // The user is free to create queues with out-of-order execution enabled
    default_queue_handler = clQueueHandler(context, devices);

    kernels = std::make_shared<clKernelMap>(context, kernels_search_path);
  }

  clWrapper::clWrapper(const clWrapper &other) {
    platform = other.platform;
    context = other.context;
    default_device = other.default_device;
    default_queue = other.default_queue;
    default_queue_handler = other.default_queue_handler;
    kernels = other.kernels;
  }

  std::unique_ptr<clWrapper>
  clWrapper::makeDefaultWrapper(std::filesystem::path kernels_search_path) {
    auto cpu_platforms = findCPUPlatform();
    auto gpu_platforms = findGPUPlatform();

    if (cpu_platforms.empty()) { throw std::runtime_error("No CPU platform found"); }

    if (not gpu_platforms.empty()) {
      return std::make_unique<clWrapper>(gpu_platforms[0], kernels_search_path);
    } else {
      return std::make_unique<clWrapper>(cpu_platforms[0], kernels_search_path);
    }
  }

  clQueueHandler clWrapper::makeQueueHandler(cl::QueueProperties properties) {
    return clQueueHandler(context, devices, properties);
  }

  cl::Program clWrapper::getProgram(const std::string &program_name) {
    std::shared_lock<std::shared_mutex> lock(main_mutex);
    auto it = programs.find(program_name);

    // If we failed to find the program in the map, we need to build it
    if (it == programs.end()) {
      // Reacquire an exclusive lock
      lock.unlock();
      std::unique_lock<std::shared_mutex> lock2(main_mutex);
      it = programs.find(program_name);

      // Check again that the program was not added while we were waiting for the lock
      if (it != programs.end()) { return it->second; }

      std::string source = readFile(program_name);
      cl::Program program(context, source);
      try {
        program.build();
        it = programs.emplace(program_name, program).first;
        return it->second;
      } catch (std::exception &e) {
        std::string build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device);
        tscl::logger(build_log, tscl::Log::Error);
        tscl::logger("Failed to build program: " + program_name, tscl::Log::Fatal);
      }
    }
    return it->second;
  }

  cl::Kernel clWrapper::getKernel(const std::string &program_name, const std::string &kernel_name) {
    cl::Program program = getProgram(program_name);
    cl::Kernel kernel(program, kernel_name.c_str());
    return kernel;
  }
}   // namespace utils