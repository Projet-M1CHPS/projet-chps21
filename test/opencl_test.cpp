#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS 1
#include <CL/opencl.hpp>
#include <clblast.h>
#include <iostream>
#include <vector>

cl::Platform getPlatform() {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  for (size_t i = 0; i < platforms.size(); i++) {
    std::cout << i << ". " << platforms[i].getInfo<CL_PLATFORM_NAME>() << std::endl;
    std::vector<cl::Device> devices;
    platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);
    for (size_t j = 0; j < devices.size(); j++) {
      std::cout << " \t" << j << " - " << devices[j].getInfo<CL_DEVICE_NAME>() << std::endl;
    }
  }

  int platform_id = -1;
  std::cout << "Select platform: ";
  while (platform_id >= platforms.size() or platform_id < 0) { std::cin >> platform_id; }
  return platforms[platform_id];
}

int main() {
  std::cout << "Beginning OpenCL test" << std::endl;
  auto platform = getPlatform();

  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

  cl::Context context(devices);
  cl::CommandQueue queue(context, devices[0]);
  size_t n = 128, m = 1024;

  cl::Buffer mat_a(context, CL_MEM_READ_WRITE, sizeof(float) * n * m);
  cl::Buffer mat_b(context, CL_MEM_READ_WRITE, sizeof(float) * n * m);
  cl::Buffer mat_c(context, CL_MEM_READ_WRITE, sizeof(float) * n * m);

  clblast::Gemm<float>(clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo,
                       n, m, m, 1.f, mat_a(), 0, m, mat_b(), 0, m, 1.f, mat_c(), 0, m, &queue());

  return 0;
}