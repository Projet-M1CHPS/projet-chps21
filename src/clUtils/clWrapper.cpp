#include "clWrapper.hpp"
#include <fstream>
#include <ncurses.h>

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
        if (devices.empty()) continue;
        std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
        for (size_t i = 0; i < devices.size(); i++)
          std::cout << "\t d" << i << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
        if (!devices.empty()) { cpu_platforms.push_back(platform); }
      }
      return cpu_platforms;
    }

    std::vector<cl::Platform> findGPUPlatform() {
      std::vector<cl::Platform> platforms;
      cl::Platform::get(&platforms);

      std::vector<cl::Platform> gpu_platforms;
      for (auto &platform : platforms) {
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) continue;
        std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
        for (size_t i = 0; i < devices.size(); i++)
          std::cout << "\t d" << i << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
        if (!devices.empty()) { gpu_platforms.push_back(platform); }
      }
      return gpu_platforms;
    }

    // Loading an OpenCL program requires it to be loaded into a buffer first
    std::string readFile(const std::string &path) {
      std::ifstream t(path);
      if (!t.is_open()) { throw std::runtime_error("Could not open file: " + path); }
      std::string res((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
      return res;
    }
  }   // namespace

  clWrapper::clWrapper(cl::Platform &platform, size_t device_id,
                       const std::filesystem::path &kernels_search_path) {
    this->platform = platform;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    std::cout << "Selected Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    for (size_t i = 0; i < devices.size(); i++) {
      if (i == 0)
        std::cout << "Default Device d0: " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
      else
        std::cout << "\t d" << i << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
    }

    default_device = devices[device_id];
    context = cl::Context(devices);
    default_queue = cl::CommandQueue(context, default_device);
    // By default, we do not enable out-of-order execution for the queue handler
    // The user is free to create queues with out-of-order execution enabled
    default_queue_handler = clQueueHandler(context, devices);

    kernels = std::make_shared<clKernelMap>(context, kernels_search_path);
  }

  clWrapper::clWrapper(const clWrapper &other) {
    // Copy everything except the mutex
    platform = other.platform;
    context = other.context;
    default_device = other.default_device;
    default_queue = other.default_queue;
    default_queue_handler = other.default_queue_handler;
    // No need to lock the mutex here, since we're just copying the pointers
    kernels = other.kernels;
  }

  clQueueHandler clWrapper::makeQueueHandler(cl::QueueProperties properties) {
    return clQueueHandler(context, devices, properties);
  }


  std::unique_ptr<clWrapper>
  clWrapper::makeDefault(const std::filesystem::path &kernels_search_path) {
    auto cpu_platforms = findCPUPlatform();
    auto gpu_platforms = findGPUPlatform();

    if (cpu_platforms.empty() and gpu_platforms.empty()) {
      throw std::runtime_error("No OpenCL platforms found");
    }

    if (not gpu_platforms.empty()) {
      return std::make_unique<clWrapper>(gpu_platforms[0], kernels_search_path);
    } else {
      return std::make_unique<clWrapper>(cpu_platforms[0], kernels_search_path);
    }
  }
}   // namespace utils