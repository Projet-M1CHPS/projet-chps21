#include "clWrapper.hpp"
#include <fstream>

namespace utils {

  namespace {
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
  }   // namespace

  clWrapper::clWrapper(clWrapper &other) {
    std::shared_lock<std::shared_mutex> lock(other.main_mutex);

    context = other.context;
    default_queue = other.default_queue;
    programs = other.programs;
    platform = other.platform;
  }

  clWrapper clWrapper::makeDefaultWrapper() {
    auto cpu_platforms = findCPUPlatform();
    auto gpu_platforms = findGPUPlatform();


    for (auto &platform : cpu_platforms) {
      std::vector<cl::Device> devices;
      platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
      for (auto &device : devices) {
        std::cout << "OpenCL CPU device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
      }
    }

    for (auto &platform : gpu_platforms) {
      std::vector<cl::Device> devices;
      platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
      for (auto &device : devices) {
        std::cout << "OpenCL CPU device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
      }
    }
    cl::Platform default_platform = cpu_platforms[0];
    /*
    if (not gpu_platforms.empty()) default_platform = gpu_platforms[0];
    else if (not cpu_platforms.empty())
      default_platform = cpu_platforms[0];
    else
      throw std::runtime_error("No OpenCL platform found");
      */

    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.empty()) {
      tscl::logger("No OpenCL devices found", tscl::Log::Error);
      throw std::runtime_error("No OpenCL devices found");
    }

    cl::Device default_device = all_devices[0];
    cl::Context context(all_devices);
    cl::CommandQueue queue(context, default_device);

    clWrapper res;
    res.context = context;
    res.default_queue = queue;
    res.default_device = default_device;
    res.platform = default_platform;
    return res;
  }


  std::string readFile(const std::string &path) {
    std::ifstream t(path);
    if (!t.is_open()) { throw std::runtime_error("Could not open file: " + path); }
    std::string res((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    return res;
  }

  cl::Program clWrapper::getProgram(const std::string &program_name) {
    std::shared_lock<std::shared_mutex> lock(main_mutex);
    auto it = programs.find(program_name);

    // If we failed to find the program in the map, we need to build it
    if (it == programs.end()) {
      // We unlock the shared mutex and reacquire it as a unique lock for the build
      lock.unlock();
      std::unique_lock<std::shared_mutex> lock2(main_mutex);
      it = programs.find(program_name);

      std::string source = readFile(program_name);
      cl::Program program(context, source);
      try {
        program.build();
        it = programs.emplace(program_name, program).first;
      } catch (std::exception &e) {

        std::string build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device);
        tscl::logger(build_log, tscl::Log::Error);
        tscl::logger("Failed to build program: " + program_name, tscl::Log::Fatal);
      }
    }
    return it->second;
  }

  void printAvailablePlatforms() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for (auto &platform : platforms) {
      std::string platform_name;
      platform.getInfo(CL_PLATFORM_NAME, &platform_name);
      std::cout << platform_name << std::endl;

      std::vector<cl::Device> devices;
      platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
      for (auto &device : devices) {
        std::string device_name;
        device.getInfo(CL_DEVICE_NAME, &device_name);
        std::cout << "\t" << device_name << std::endl;
      }
    }
  }
}   // namespace utils