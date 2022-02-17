#pragma once
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS 1
#include "Matrix.hpp"
#include <CL/opencl.hpp>

namespace math {

  /**
   * @brief Wrapper class for Matrix stored using OpenCL
   * If not specified, the matrix is allocated on the default context
   */
  class clMatrix {
  public:
    clMatrix() = default;
    clMatrix(const clMatrix &) = default;
    clMatrix(clMatrix &&) = default;

    clMatrix &operator=(const clMatrix &) = default;
    clMatrix &operator=(clMatrix &&) = default;

    /**
     * @brief allocates a new matrix on the device
     * @param rows Numbers of rows of the new matrix
     * @param cols Numbers of columns of the new matrix
     * @param context OpenCL context to use
     */
    clMatrix(size_t rows, size_t cols, const cl::Context &context = cl::Context::getDefault())
        : rows(rows), cols(cols) {
      data = cl::Buffer(context, CL_MEM_READ_WRITE, rows * cols * sizeof(float));
    }

    /**
     * @brief allocates a new matrix on the device
     * @param source Source ptr to copy from
     * @param rows Numbers of rows of the new matrix
     * @param cols Numbers of columns of the new matrix
     * @param context OpenCL context to use
     */
    clMatrix(float *source, size_t rows, size_t cols,
             const cl::Context &context = cl::Context::getDefault())
        : rows(rows), cols(cols) {
      data = cl::Buffer(context, CL_MEM_READ_WRITE, rows * cols * sizeof(float), source);
    }

    explicit clMatrix(const math::FloatMatrix &matrix,
                      const cl::Context &context = cl::Context::getDefault())
        : rows(matrix.getRows()), cols(matrix.getCols()) {
      data = cl::Buffer(context, CL_MEM_READ_WRITE, rows * cols * sizeof(float),
                        (void *) matrix.getData());
    }

    cl::Buffer &getBuffer() { return data; }
    const cl::Buffer &getBuffer() const { return data; }

    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }

  private:
    cl::Buffer data;
    size_t rows = 0, cols = 0;
  };

  /**
   * @brief Fetch an open cl matrix an store it in a matrix in the host memory
   * @param matrix The matrix stored in device memory to fetch
   * @param queue The queue to use for the fetch
   * @return A copy of the matrix in the host memory
   */
  math::Matrix<float> fetchClMatrix(const clMatrix &matrix, const cl::CommandQueue &queue);
}   // namespace math