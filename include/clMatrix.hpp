#pragma once
#define CL_HPP_ENABLE_EXCEPTIONS 1
#include <CL/opencl.hpp>

namespace math {

  /**
   * @brief Wrapper class for Matrix stored using OpenCL
   * If not specified, the matrix is allocated on the default context
   */
  class clMatrix {
  public:
    clMatrix() = default;

    /**
     * @brief allocates a new matrix on the device
     * @param rows Numbers of rows of the new matrix
     * @param cols Numbers of columns of the new matrix
     * @param context OpenCL context to use
     * @param queue OpenCL command queue to use
     */
    clMatrix(size_t rows, size_t cols, cl::Context context = cl::Context::getDefault(),
             cl::CommandQueue queue = cl::CommandQueue::getDefault())
        : rows(rows), cols(cols) {
      data = cl::Buffer(context, CL_MEM_READ_WRITE, rows * cols * sizeof(float));
    }

    /**
     * @brief allocates a new matrix on the device
     * @param source Source ptr to copy from
     * @param rows Numbers of rows of the new matrix
     * @param cols Numbers of columns of the new matrix
     * @param context OpenCL context to use
     * @param queue OpenCL command queue to use
     */
    clMatrix(float *source, size_t rows, size_t cols,
             const cl::Context &context = cl::Context::getDefault(),
             const cl::CommandQueue &queue = cl::CommandQueue::getDefault())
        : rows(rows), cols(cols) {
      data = cl::Buffer(context, CL_MEM_READ_WRITE, rows * cols * sizeof(float), NULL);
    }

    /**
     * @brief allocates a new matrix on the device
     * @param source Source char array to copy from
     * @param rows Numbers of rows of the new matrix
     * @param cols Numbers of columns of the new matrix
     * @param context OpenCL context to use
     * @param queue OpenCL command queue to use
     */
    clMatrix(char *source, size_t rows, size_t cols,
             const cl::Context &context = cl::Context::getDefault(),
             const cl::CommandQueue &queue = cl::CommandQueue::getDefault())
        : rows(rows), cols(cols) {
      data = cl::Buffer(context, CL_MEM_READ_WRITE, rows * cols * sizeof(float));
    }

    cl::Buffer &getBuffer() { return data; }

    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }

  private:
    cl::Buffer data;
    size_t rows = 0, cols = 0;
  };
}   // namespace math