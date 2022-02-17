#include "clWrapper.hpp"
#include <fstream>

namespace utils {

  namespace {

    // We use a global variable to store the default wrapper
    // Normally I would implement a singleton using a static variable
    // However, this allows one to swap the wrapper at runtime
    //
    // Since wrappers are copyable, it is safe to swap the wrapper while it's in use elsewhere
    // since OpenCL will only deallocate platforms, devices, ..., when all references are destroyed
    // To ensure this, we need to ensure the default wrapper is only retrievable through copy
    std::unique_ptr<clWrapper> default_wrapper;

    // Needed to prevent the wrapper being swapped while it is being copied
    std::shared_mutex default_wrapper_mutex;

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

    // Takes an exclusive lock on the default wrapper to prevent erroneous use
    // This ensures that the caller
    void makeDefaultWrapper() {
      std::scoped_lock<std::shared_mutex> lock(default_wrapper_mutex);
      if (default_wrapper) return;

      std::vector<cl::Platform> cpu_platforms = findCPUPlatform();
      default_wrapper = std::make_unique<clWrapper>(cpu_platforms[0]);
    }

  }   // namespace

  clWrapper::clWrapper(cl::Platform &platform) {
    this->platform = platform;
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    default_device = devices[0];
    context = cl::Context(devices);
    default_queue = cl::CommandQueue(context, default_device);
    // By default, we do not enable out-of-order execution for the queue handler
    // The user is free to create queues with out-of-order execution enabled
    default_queue_handler = clQueueHandler(context, devices);
  }

  clWrapper::clWrapper(clWrapper &other) {
    context = other.context;
    default_queue = other.default_queue;
    platform = other.platform;

    // We lock the mutex to ensure no program is added while we are copying them
    std::shared_lock<std::shared_mutex> lock(other.main_mutex);
    // No issues copying the programs, as they are only wrappers around cl objects
    programs = other.programs;
  }

  clWrapper clWrapper::getDefaultWrapper() {
    // We use a shared_mutex to avoid contentions
    std::shared_lock<std::shared_mutex> lock(default_wrapper_mutex);

    // Bloat code to ensure thread safety
    // This will only occur the first time the default wrapper is allocated
    if (not default_wrapper) {
      // If the default_wrapper is not allocated yet, we need to free the shared_lock and acquire an
      // exclusive lock
      // this is inefficient, but guarantees thread safety and should only be done for the first
      // allocation
      lock.unlock();
      makeDefaultWrapper();
      lock.lock();
    }
    // Copy the default wrapper while we own the shared lock
    return *default_wrapper;
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