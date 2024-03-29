#include "clKernelMap.hpp"
#include "Logger.hpp"
#include <fstream>
#include <utility>

namespace fs = std::filesystem;

namespace utils {

  clKernelMap::clKernelMap(cl::Context &context, std::filesystem::path kernels_path)
      : context(context), search_path(std::move(kernels_path)) {}

  cl::Program &clKernelMap::getProgram(const std::string &program_name) {
    std::shared_lock<std::shared_mutex> lock(map_mutex);
    auto it = map.find(program_name);
    if (it != map.end()) return it->second;

    // reacquire an exclusive lock
    lock.unlock();
    std::scoped_lock<std::shared_mutex> exclusive_lock(map_mutex);

    fs::path path = search_path / program_name;
    if (not fs::exists(path)) {
      throw std::runtime_error("Could not find program " + program_name);
    }

    // read the program and compile it
    std::ifstream file(path);
    std::string program_str((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    auto res = map.emplace(program_name, cl::Program(context, program_str));
    try {
      res.first->second.build();
    } catch (const cl::Error &e) {
      tscl::logger("Error building program " + program_name + ": " + e.what(), tscl::Log::Error);
      tscl::logger("Build log:\n " + res.first->second.getBuildInfo<CL_PROGRAM_BUILD_LOG>(
                                             context.getInfo<CL_CONTEXT_DEVICES>()[0]),
                   tscl::Log::Fatal);
    }
    return res.first->second;
  }

  cl::Kernel clKernelMap::getKernel(const std::string &program_name,
                                    const std::string &kernel_name) {
    auto &program = getProgram(program_name);
    cl::Kernel res = {program, kernel_name.c_str()};
    return res;
  }
}   // namespace utils