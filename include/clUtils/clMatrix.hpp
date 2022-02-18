#pragma once
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS 1
#include "Matrix.hpp"
#include "clWrapper.hpp"
#include <CL/opencl.hpp>

namespace math {

  /**
   * @brief Wrapper class for Matrix stored using OpenCL
   * If not specified, the matrix is allocated on the default context
   */
  class clMatrix {
  public:
    clMatrix() = default;

    // clMatrix are not copyable without a context
    // We delete the copy operator to prevent misuse
    clMatrix(const clMatrix &) = delete;
    clMatrix(clMatrix &&) = default;
    clMatrix &operator=(clMatrix &&) = default;

    /**
     * @brief allocates a new matrix on the device
     * @param rows Numbers of rows of the new matrix
     * @param cols Numbers of columns of the new matrix
     * @param context OpenCL context to use
     */
    clMatrix(size_t rows, size_t cols, utils::clWrapper &wrapper) : rows(rows), cols(cols) {
      data = cl::Buffer(wrapper.getContext(), CL_MEM_READ_WRITE, rows * cols * sizeof(float));
    }

    /**
     * @brief allocates a new matrix on the device
     * @param source Source ptr to copy from
     * @param rows Numbers of rows of the new matrix
     * @param cols Numbers of columns of the new matrix
     * @param context OpenCL context to use
     */
    clMatrix(float *source, size_t rows, size_t cols, utils::clWrapper &wrapper)
        : rows(rows), cols(cols) {
      data = cl::Buffer(wrapper.getContext(), CL_MEM_READ_WRITE, rows * cols * sizeof(float),
                        source);
    }

    explicit clMatrix(const math::FloatMatrix &matrix, utils::clWrapper &wrapper)
        : rows(matrix.getRows()), cols(matrix.getCols()) {
      data = cl::Buffer(wrapper.getContext(), CL_MEM_READ_WRITE, rows * cols * sizeof(float),
                        (void *) matrix.getData());
    }

    void fromFloatMatrix(const math::FloatMatrix &matrix, utils::clWrapper &wrapper) {
      if (rows * cols != matrix.getRows() * matrix.getCols()) {
        data = cl::Buffer(wrapper.getContext(), CL_MEM_READ_WRITE, rows * cols * sizeof(float),
                          (void *) matrix.getData());
      } else {
        wrapper.getDefaultQueue().enqueueWriteBuffer(data, CL_TRUE, 0, rows * cols * sizeof(float),
                                                     (void *) matrix.getData());
      }
    }

    FloatMatrix toFloatMatrix(utils::clWrapper &wrapper) {
      FloatMatrix matrix(rows, cols);
      wrapper.getDefaultQueue().enqueueReadBuffer(data, CL_TRUE, 0, rows * cols * sizeof(float),
                                                  (void *) matrix.getData());
      return matrix;
    }

    cl::Buffer &getBuffer() { return data; }
    const cl::Buffer &getBuffer() const { return data; }

    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }

  private:
    cl::Buffer data;
    size_t rows = 0, cols = 0;
  };
}   // namespace math