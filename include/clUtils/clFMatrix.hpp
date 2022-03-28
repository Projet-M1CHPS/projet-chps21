#pragma once
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS 1
#include "Matrix.hpp"
#include "clWrapper.hpp"
#include <CL/opencl.hpp>
#include <clblast.h>

namespace math {

  /**
   * @brief Wrapper class for a float Matrix stored using OpenCL
   *
   * This class does not hold a reference to the context (To reduce the memory footprint) used for
   * the buffer, so it is the responsibility of the user to ensure that the context used for the
   * buffer is the same as the one used for the computation
   */
  class clFMatrix {
  public:
    clFMatrix() = default;
    clFMatrix &operator=(const clFMatrix &other);
    clFMatrix &operator=(const FloatMatrix &other);

    clFMatrix(clFMatrix &&) = default;
    clFMatrix &operator=(clFMatrix &&) = default;

    /**
     * @brief Allocates a new matrix on the platform
     * @param rows Numbers of rows of the new matrix
     * @param cols Numbers of columns of the new matrix
     * @param wrapper Wrapper to be used for memory allocation
     */
    clFMatrix(size_t rows, size_t cols);

    /**
     * @brief Allocates a new matrix on the device, and copy the content of the host array to it
     * @param source The source float array to copy from
     * @param rows The numbers of rows of the new matrix
     * @param cols The numbers of cols of the new matrix
     * @param wrapper The wrapper to be used for the matrix creation
     * @param queue The queue to be used for the copy
     * @param blocking True if the operation is blocking, false otherwise
     * Note that if the operation is non-blocking, the user is responsible for ensuring that the
     * host array remains valid until the operation is finished
     */
    clFMatrix(const float *source, size_t rows, size_t cols, cl::CommandQueue &queue,
              bool blocking = true);

    /**
     * @brief Allocates a new matrix on the device and copies the data from the host, using the
     * default queue
     * @param source Source ptr to copy from
     * @param rows Numbers of rows of the new matrix
     * @param cols Numbers of columns of the new matrix
     * @param context OpenCL context to use
     */
    clFMatrix(const float *source, size_t rows, size_t cols, bool blocking = true)
        : clFMatrix(source, rows, cols, utils::cl_wrapper.getDefaultQueue(), blocking) {}

    /**
     * @brief Copies a FloatMatrix to the device
     * @param matrix the matrix to copy
     * @param wrapper The wrapper to used for memory allocation
     * @param queue The queue to be used for the copy operation
     * @param blocking True if the operation is blocking, false otherwise
     * Note that if the operation is non-blocking, the user is responsible for ensuring that the
     * matrix remains valid until the operation is finished
     */
    clFMatrix(const math::FloatMatrix &matrix, cl::CommandQueue &queue, bool blocking = true);

    /**
     * @brief Copies a FloatMatrix to the device, using the default queue
     * @param matrix The matrix to copy
     * @param wrapper The wrapper to used for memory allocation
     * @param blocking  True if the operation is blocking, false otherwise
     */
    clFMatrix(const math::FloatMatrix &matrix, bool blocking = true)
        : clFMatrix(matrix, utils::cl_wrapper.getDefaultQueue(), blocking) {}


    /**
     * @brief Copies a matrix on the device
     * @param other The matrix to copy
     * @param wrapper The wrapper to be used for memory allocation
     * @param queue The queue to use for the copy operation
     * @param blocking True if the operation is blocking, false otherwise
     */
    clFMatrix(const clFMatrix &other, cl::CommandQueue &queue, bool blocking = true);

    /**
     * @brief Copies a matrix on the device, using the default queue
     * @param other The matrix to copy
     * @param wrapper The wrapper to use for memory allocation
     * @param blocking True if the operation is blocking, false otherwise
     */
    clFMatrix(const clFMatrix &other, bool blocking = true)
        : clFMatrix(other, utils::cl_wrapper.getDefaultQueue(), blocking) {}

    cl::Buffer &getBuffer() { return data; }
    [[nodiscard]] const cl::Buffer &getBuffer() const { return data; }

    [[nodiscard]] size_t getRows() const { return rows; }
    [[nodiscard]] size_t getCols() const { return cols; }
    [[nodiscard]] size_t size() const { return rows * cols; }

    /**
     * @brief Copies a matrix on the host to the device, replacing the current matrix
     * @param matrix The matrix to copy
     * @param wrapper The wrapper to use for memory allocation if needed
     * @param queue The queue to use for the copy operation
     * @param blocking  True if the operation is blocking, false otherwise
     * Note that if this operation is non-blocking, the user is responsible for ensuring that the
     * matrix remains valid until the operation is finished
     */
    void fromFloatMatrix(const math::FloatMatrix &matrix, cl::CommandQueue &queue,
                         bool blocking = true);

    /**
     * @brief Copies a matrix on the host to the device, replacing the current matrix, using the
     * default queue
     * @param matrix The matrix to copy
     * @param wrapper The wrapper to use for memory allocation if needed
     * @param blocking  True if the operation is blocking, false otherwise
     * Note that if this operation is non-blocking, the user is responsible for ensuring that the
     * matrix remains valid until the operation is finished
     */
    void fromFloatMatrix(const math::FloatMatrix &matrix, bool blocking = true) {
      fromFloatMatrix(matrix, utils::cl_wrapper.getDefaultQueue(), blocking);
    }

    /**
     * @brief Copies the matrix on the device to a matrix on the host
     * @param wrapper The wrapper to use for memory allocation if needed
     * @param queue The queue to use for the copy operation
     * @param blocking True if the operation is blocking, false otherwise
     * Note that if this operation is non-blocking, the user is responsible for ensuring that the
     * operation is finished before using the matrix
     * @return The new matrix
     */
    [[nodiscard]] FloatMatrix toFloatMatrix(cl::CommandQueue &queue, bool blocking = true) const;

    /**
     * @brief Copies the matrix on the device to a matrix on the host using the default queue
     * @param wrapper The wrapper to use for memory allocation if needed
     * @param blocking True if the operation is blocking, false otherwise
     * Note that if this operation is non-blocking, the user is responsible for ensuring that the
     * operation is finished before using the matrix
     * @return The new matrix
     */
    [[nodiscard]] FloatMatrix toFloatMatrix(bool blocking = true) const {
      return toFloatMatrix(utils::cl_wrapper.getDefaultQueue(), blocking);
    }

    /**
     * @brief Sum the element of the matrix an return the result. This operation is always blocking
     * @param wrapper The wrapper to use for this operation
     * @param queue The queue to use for this operation
     * @return The sum of the elements of the matrix
     */
    [[nodiscard]] float sumReduce(cl::CommandQueue &queue) const;

    /**
     * @brief Sum the element of the matrix an return the result. This operation is always blocking,
     * and uses the default queue
     * @param wrapper The wrapper to use for this operation
     * @return The sum of the elements of the matrix
     */
    [[nodiscard]] float sumReduce() const { return sumReduce(utils::cl_wrapper.getDefaultQueue()); }

    /**
     * @brief Sum the element of the matrix an return the result. This operation is always blocking
     * @param wrapper  The wrapper to use for this operation
     * @param queue The queue to use for this operation
     * @return The l2 norm of the matrix
     */
    [[nodiscard]] float l2norm(cl::CommandQueue &queue) const;

    /**
     * @brief Sum the element of the matrix an return the result. This operation is always blocking,
     * and uses the default queue
     * @param wrapper  The wrapper to use for this operation
     * @return The l2 norm of the matrix
     */
    [[nodiscard]] float l2norm() const { return l2norm(utils::cl_wrapper.getDefaultQueue()); }

    [[nodiscard]] clFMatrix transpose(cl::CommandQueue &queue, bool blocking = false) const;

    [[nodiscard]] clFMatrix transpose(bool blocking = false) const {
      return transpose(utils::cl_wrapper.getDefaultQueue(), blocking);
    }

    /**
     * @brief Inplace addition of two matrices. By default, this operation is non-blocking
     * @param other The other matrix to add
     * @param wrapper The wrapper of the platform
     * @param queue The queue to use for this operation
     * @param blocking True if the operation is blocking, false otherwise
     * Note that if this operation is non-blocking, the user is responsible for ensuring that the
     * operation is finished before using the matrix
     * @return This matrix after the addition
     */
    void ipadd(float factor, const clFMatrix &other, cl::CommandQueue &queue,
               bool blocking = false);

    /**
     * @brief Inplace addition of two matrices. By default, this operation is non-blocking, and uses
     the default queue.
     * @param other The other matrix to add
     * @param wrapper The wrapper of the platform
     * @param blocking True if the operation is blocking, false otherwise
     * Note that if this operation is non-blocking, the user is responsible for ensuring that the
     * operation is finished before using the matrix
     * @return This matrix after the addition
     */
    void ipadd(float factor, const clFMatrix &other, bool blocking = false) {
      ipadd(factor, other, utils::cl_wrapper.getDefaultQueue(), blocking);
    }

    /**
     * @brief Adds two matrices. By default, this operation is non-blocking
     * @param other The other matrix to add
     * @param wrapper The wrapper to use for memory allocation
     * @param queue The queue to use for this operation
     * @param blocking True if the operation is blocking, false otherwise
     * Note that if this operation is non-blocking, the user is responsible for ensuring that the
     * operation is finished before using the matrix
     * @return this + other
     */
    [[nodiscard]] clFMatrix add(float factor, const clFMatrix &other, cl::CommandQueue &queue,
                                bool blocking = false) const;

    /**
     * @brief Adds two matrices. By default, this operation is non-blocking, and uses the default
     * queue
     * @param other The other matrix to add
     * @param wrapper The wrapper to use for memory allocation
     * @param blocking True if the operation is blocking, false otherwise
     * Note that if this operation is non-blocking, the user is responsible for ensuring that the
     * operation is finished before using the matrix
     * @return this + other
     */
    [[nodiscard]] clFMatrix add(float factor, const clFMatrix &other, bool blocking = false) const {
      return add(factor, other, utils::cl_wrapper.getDefaultQueue(), blocking);
    }

    /**
     * @brief Inplace subtraction of two matrices. By default, this operation is non-blocking
     * @param other The other matrix to subtract
     * @param wrapper The wrapper of the platform
     * @param queue The queue to use for this operation
     * @param blocking True if the operation is blocking, false otherwise
     * Note that if this operation is non-blocking, the user is responsible for ensuring that the
     * operation is finished before using the matrix
     */
    void ipsub(float factor, const clFMatrix &other, cl::CommandQueue &queue,
               bool blocking = false);

    /**
     * @brief Inplace subtraction of two matrices. By default, this operation is non-blocking and
     * uses the default queue
     * @param other The other matrix to subtract
     * @param wrapper The wrapper of the platform
     * @param blocking True if the operation is blocking, false otherwise
     * Note that if this operation is non-blocking, the user is responsible for ensuring that the
     * operation is finished before using the matrix
     */
    void ipsub(float factor, const clFMatrix &other, bool blocking = false) {
      ipsub(factor, other, utils::cl_wrapper.getDefaultQueue(), blocking);
    }

    /**
     * @brief Subtracts two matrices. By default, this operation is non-blocking
     * @param other The other matrix to subtract
     * @param wrapper The wrapper of the platform
     * @param queue The queue to use for this operation
     * @param blocking True if the operation is blocking, false otherwise
     * Note that if this operation is non-blocking, the user is responsible for ensuring that the
     * operation is finished before using the matrix
     * @return this - other
     */
    [[nodiscard]] clFMatrix sub(float factor, const clFMatrix &other, cl::CommandQueue &queue,
                                bool blocking = false) const;

    /**
     * @brief Subtracts two matrices. By default, this operation is non-blocking and uses the
     * default queue
     * @param other The other matrix to subtract
     * @param wrapper The wrapper of the platform
     * @param blocking True if the operation is blocking, false otherwise
     * Note that if this operation is non-blocking, the user is responsible for ensuring that the
     * operation is finished before using the matrix
     * @return this - other
     */
    [[nodiscard]] clFMatrix sub(float factor, const clFMatrix &other, bool blocking = false) const {
      return sub(factor, other, utils::cl_wrapper.getDefaultQueue(), blocking);
    }

    void ipscale(float scale, cl::CommandQueue &queue, bool blocking = false);

    void ipscale(float scale, bool blocking = false) {
      ipscale(scale, utils::cl_wrapper.getDefaultQueue(), blocking);
    }

    [[nodiscard]] clFMatrix scale(float scale, cl::CommandQueue &queue,
                                  bool blocking = false) const;

    [[nodiscard]] clFMatrix scale(float s, bool blocking = false) const {
      return scale(s, utils::cl_wrapper.getDefaultQueue(), blocking);
    }

    void iphadamard(const clFMatrix &other, cl::CommandQueue &queue, bool blocking = false) const;

    void iphadamard(const clFMatrix &other, bool blocking = false) const {
      iphadamard(other, utils::cl_wrapper.getDefaultQueue(), blocking);
    }

    clFMatrix hadamard(const clFMatrix &other, cl::CommandQueue &queue,
                       bool blocking = false) const;
    clFMatrix hadamard(const clFMatrix &other, bool blocking = false) const {
      return hadamard(other, utils::cl_wrapper.getDefaultQueue(), blocking);
    }


    [[nodiscard]] static clFMatrix gemm(float alpha, bool transpose_a, const clFMatrix &A,
                                        bool transpose_b, const clFMatrix &B,
                                        cl::CommandQueue &queue, bool blocking = false);

    [[nodiscard]] static clFMatrix gemm(float alpha, bool transpose_a, const clFMatrix &A,
                                        bool transpose_b, const clFMatrix &B,
                                        bool blocking = false) {
      return gemm(alpha, transpose_a, A, transpose_b, B, utils::cl_wrapper.getDefaultQueue(),
                  blocking);
    }

    [[nodiscard]] static clFMatrix gemm(float alpha, bool transpose_a, const clFMatrix &A,
                                        bool transpose_b, const clFMatrix &B, float beta,
                                        const clFMatrix &C, cl::CommandQueue &queue,
                                        bool blocking = false);

    [[nodiscard]] static clFMatrix gemm(float alpha, bool transpose_a, const clFMatrix &A,
                                        bool transpose_b, const clFMatrix &B, float beta,
                                        const clFMatrix &C, bool blocking = false) {
      return gemm(alpha, transpose_a, A, transpose_b, B, beta, C,
                  utils::cl_wrapper.getDefaultQueue(), blocking);
    }

  private:
    cl::Buffer data;
    size_t rows = 0, cols = 0;
  };
}   // namespace math