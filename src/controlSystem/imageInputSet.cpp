#include "imageInputSet.hpp"
#include <stack>

namespace fs = std::filesystem;
using namespace math;
using namespace utils;

namespace control {

  namespace {


    clFMatrix loadFile(const fs::path &path, clWrapper &wrapper, cl::CommandQueue &queue,
                       bool blocking) {
      auto image = image::ImageSerializer::load(path);
      // Load the conversion kernel
      cl::Kernel kernel =
              wrapper.getKernels().getKernel("NormalizeCharToFloat.cl", "normalizeCharToFloat");

      // Allocate a new cl matrix
      math::clFMatrix res(image.getWidth() * image.getHeight(), 1);

      // We need to load the image to the cl device before applying the kernel
      cl::Buffer img_buffer(wrapper.getContext(), CL_MEM_READ_ONLY,
                            image.getWidth() * image.getHeight());

      queue.enqueueWriteBuffer(img_buffer, CL_FALSE, 0, image.getSize(), image.getData());

      kernel.setArg(0, img_buffer);
      kernel.setArg(1, res.getBuffer());

      // We also normalize the matrix by a given factor
      kernel.setArg(2, 255);
      // OpenCL does not support size_t, so we cast it to unsigned long
      kernel.setArg(3, (cl_ulong) image.getSize());

      queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image.getSize()),
                                 cl::NullRange);
      return res;
    }
  }   // namespace

  ImageInputSet ImageInputSet::load(const fs::path &source, clWrapper &wrapper,
                                    cl::CommandQueue &queue, bool blocking) {
    if (not fs::exists(source)) {
      throw std::runtime_error("ImageInputSet::fromDirectory: Source directory does not exist");
    }

    ImageInputSet res;

    if (fs::is_directory(source)) {
      std::stack<fs::path> directories;
      directories.push(source);

      while (not directories.empty()) {
        auto &dir = directories.top();
        for (const auto &entry : fs::directory_iterator(dir)) {
          if (fs::is_directory(entry)) {
            directories.push(entry);
          } else if (fs::is_regular_file(entry)) {
            res.append(loadFile(entry, wrapper, queue, blocking));
          }
        }
        directories.pop();
      }

    } else {
      res.append(loadFile(source, wrapper, queue, blocking));
    }
    return res;
  }
}   // namespace control
