#include "clWrapper.hpp"
#include <fstream>
#include <ncurses.h>

namespace fs = std::filesystem;

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
        for (size_t i = 0; i < devices.size(); i++) {
          std::cout << "\t d" << i << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
        }
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
        for (size_t i = 0; i < devices.size(); i++) {
          std::cout << "\t d" << i << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
          std::cout << "\t Memory alignment" << devices[i].getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>()
                    << std::endl;
          std::cout << "device fission: " << devices[i].getInfo<CL_DEVICE_PARTITION_MAX_SUB_DEVICES>()
                    << std::endl;
        }
        if (!devices.empty()) { gpu_platforms.push_back(platform); }
      }
      return gpu_platforms;
    }
  }   // namespace

  clWrapper cl_wrapper;

  clWrapper::clWrapper(cl::Platform &platform, size_t device_id,
                       const std::filesystem::path &kernels_search_path) {
    fs::path absolute_kernel_path = kernels_search_path;

    if (kernels_search_path.is_relative()) {
      absolute_kernel_path =
              boost::dll::program_location().remove_filename().string() / kernels_search_path;
    }

    if (not fs::exists(absolute_kernel_path)) {
      throw std::runtime_error("Kernel path does not exist: " + absolute_kernel_path.string());
    }

    this->platform = platform;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    default_device = devices[device_id];

    context = cl::Context(devices);
    default_queue = cl::CommandQueue(context, default_device);
    // By default, we do not enable out-of-order execution for the queue handler
    // The user is free to create queues with out-of-order execution enabled

    kernels = std::make_shared<clKernelMap>(context, absolute_kernel_path);
  }

  clWrapper &clWrapper::operator=(const clWrapper &other) {
    if (this == &other) return *this;

    // Copy everything except the mutex
    platform = other.platform;
    context = other.context;
    default_device = other.default_device;
    devices = other.devices;
    default_queue = other.default_queue;
    // No need to lock the mutex here, since we're just copying the pointers
    kernels = other.kernels;
    return *this;
  }

  clWrapper &clWrapper::operator=(clWrapper &&other) noexcept {
    if (this == &other) return *this;

    // Copy everything except the mutex
    platform = other.platform;
    context = other.context;
    default_device = other.default_device;
    devices = std::move(other.devices);
    default_queue = other.default_queue;
    // No need to lock the mutex here, since we're just copying the pointers
    kernels = other.kernels;
    return *this;
  }

  std::unique_ptr<clWrapper>
  clWrapper::makeDefault(const std::filesystem::path &kernels_search_path) {
    auto cpu_platforms = findCPUPlatform();
    auto gpu_platforms = findGPUPlatform();

    if (cpu_platforms.empty() and gpu_platforms.empty()) { std::terminate(); }

    if (not gpu_platforms.empty()) {
      return std::make_unique<clWrapper>(gpu_platforms[0], kernels_search_path);
    } else {
      return std::make_unique<clWrapper>(cpu_platforms[0], kernels_search_path);
    }
  }

  clWrapper &clWrapper::initOpenCL(clWrapper &wrapper) noexcept {
    static std::atomic<bool> is_init = false;
    static std::mutex init_mutex;

    std::scoped_lock<std::mutex> lock(init_mutex);

    if (is_init) {
      std::cerr << "clWrapper::initOpenCL() has already been called.\n";
      std::terminate();
    }

    tscl::logger("Initializing OpenCL environnement...", tscl::Log::Information);

    cl_wrapper = wrapper;

    cl::Platform::setDefault(cl_wrapper.platform);
    cl::Context::setDefault(cl_wrapper.context);
    cl::Device::setDefault(cl_wrapper.default_device);
    cl::CommandQueue::setDefault(cl::CommandQueue(cl_wrapper.context, cl_wrapper.default_device));

    std::string platform_name = cl_wrapper.platform.getInfo<CL_PLATFORM_NAME>();
    std::string platform_version = cl_wrapper.platform.getInfo<CL_PLATFORM_VERSION>();
    std::string platform_vendor = cl_wrapper.platform.getInfo<CL_PLATFORM_VENDOR>();
    tscl::logger("Platform: " + platform_name + " " + platform_version + " (" + platform_vendor +
                         ")",
                 tscl::Log::Information);

    for (size_t i = 0; auto &dev : cl_wrapper.devices) {
      std::string device_name = dev.getInfo<CL_DEVICE_NAME>();
      std::string device_version = dev.getInfo<CL_DEVICE_VERSION>();
      std::string device_vendor = dev.getInfo<CL_DEVICE_VENDOR>();
      std::stringstream ss;
      ss << "Device " << i << ": " << device_name << " " << device_version;
      tscl::logger(ss.str(), tscl::Log::Information);
      i++;
    }

    tscl::logger("OpenCL environnement initialized.", tscl::Log::Information);

    is_init = true;

    return cl_wrapper;
  }

}   // namespace utils