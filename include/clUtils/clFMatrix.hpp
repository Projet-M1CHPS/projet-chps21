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

    // clFMatrix are not copyable without the context
    clFMatrix(const clFMatrix &) = delete;
    clFMatrix &operator=(const clFMatrix &) = delete;

    clFMatrix(clFMatrix &&) = default;
    clFMatrix &operator=(clFMatrix &&) = default;

    /**
     * @brief Allocates a new matrix on the platform
     * @param rows Numbers of rows of the new matrix
     * @param cols Numbers of columns of the new matrix
     * @param wrapper Wrapper to be used for memory allocation
     */
    clFMatrix(size_t rows, size_t cols, utils::clWrapper &wrapper);

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
    clFMatrix(const float *source, size_t rows, size_t cols, utils::clWrapper &wrapper,
              cl::CommandQueue &queue, bool blocking = true);

    /**
     * @brief Allocates a new matrix on the device and copies the data from the host, using the
     * default queue
     * @param source Source ptr to copy from
     * @param rows Numbers of rows of the new matrix
     * @param cols Numbers of columns of the new matrix
     * @param context OpenCL context to use
     */
    clFMatrix(const float *source, size_t rows, size_t cols, utils::clWrapper &wrapper,
              bool blocking = true)
        : clFMatrix(source, rows, cols, wrapper, wrapper.getDefaultQueue(), blocking) {}

    /**
     * @brief Copies a FloatMatrix to the device
     * @param matrix the matrix to copy
     * @param wrapper The wrapper to used for memory allocation
     * @param queue The queue to be used for the copy operation
     * @param blocking True if the operation is blocking, false otherwise
     * Note that if the operation is non-blocking, the user is responsible for ensuring that the
     * matrix remains valid until the operation is finished
     */
    clFMatrix(const math::FloatMatrix &matrix, utils::clWrapper &wrapper, cl::CommandQueue &queue,
              bool blocking = true);
    /**
     * @brief Copies a FloatMatrix to the device, using the default queue
     * @param matrix The matrix to copy
     * @param wrapper The wrapper to used for memory allocation
     * @param blocking  True if the operation is blocking, false otherwise
     */
    explicit clFMatrix(const math::FloatMatrix &matrix, utils::clWrapper &wrapper,
                       bool blocking = true)
        : clFMatrix(matrix, wrapper, wrapper.getDefaultQueue(), blocking) {}


    /**
     * @brief Copies a matrix on the device
     * @param other The matrix to copy
     * @param wrapper The wrapper to be used for memory allocation
     * @param queue The queue to use for the copy operation
     * @param blocking True if the operation is blocking, false otherwise
     */
    clFMatrix(const clFMatrix &other, utils::clWrapper &wrapper, cl::CommandQueue &queue,
              bool blocking = true);

    /**
     * @brief Copies a matrix on the device, using the default queue
     * @param other The matrix to copy
     * @param wrapper The wrapper to use for memory allocation
     * @param blocking True if the operation is blocking, false otherwise
     */
    clFMatrix(const clFMatrix &other, utils::clWrapper &wrapper, bool blocking = true)
        : clFMatrix(other, wrapper, wrapper.getDefaultQueue(), blocking) {}

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
    void fromFloatMatrix(const math::FloatMatrix &matrix, utils::clWrapper &wrapper,
                         cl::CommandQueue &queue, bool blocking = true);

    /**
     * @brief Copies a matrix on the host to the device, replacing the current matrix, using the
     * default queue
     * @param matrix The matrix to copy
     * @param wrapper The wrapper to use for memory allocation if needed
     * @param blocking  True if the operation is blocking, false otherwise
     * Note that if this operation is non-blocking, the user is responsible for ensuring that the
     * matrix remains valid until the operation is finished
     */
    void fromFloatMatrix(const math::FloatMatrix &matrix, utils::clWrapper &wrapper,
                         bool blocking = true) {
      fromFloatMatrix(matrix, wrapper, wrapper.getDefaultQueue(), blocking);
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
    [[nodiscard]] FloatMatrix toFloatMatrix(utils::clWrapper &wrapper, cl::CommandQueue &queue,
                                            bool blocking = true) const;

    /**
     * @brief Copies the matrix on the device to a matrix on the host using the default queue
     * @param wrapper The wrapper to use for memory allocation if needed
     * @param blocking True if the operation is blocking, false otherwise
     * Note that if this operation is non-blocking, the user is responsible for ensuring that the
     * operation is finished before using the matrix
     * @return The new matrix
     */
    [[nodiscard]] FloatMatrix toFloatMatrix(utils::clWrapper &wrapper, bool blocking = true) const {
      return toFloatMatrix(wrapper, wrapper.getDefaultQueue(), blocking);
    }

    /**
     * @brief Sum the element of the matrix an return the result. This operation is always blocking
     * @param wrapper The wrapper to use for this operation
     * @param queue The queue to use for this operation
     * @return The sum of the elements of the matrix
     */
    [[nodiscard]] float sumReduce(utils::clWrapper &wrapper, cl::CommandQueue &queue) const;

    /**
     * @brief Sum the element of the matrix an return the result. This operation is always blocking,
     * and uses the default queue
     * @param wrapper The wrapper to use for this operation
     * @return The sum of the elements of the matrix
     */
    [[nodiscard]] float sumReduce(utils::clWrapper &wrapper) const {
      return sumReduce(wrapper, wrapper.getDefaultQueue());
    }

    /**
     * @brief Sum the element of the matrix an return the result. This operation is always blocking
     * @param wrapper  The wrapper to use for this operation
     * @param queue The queue to use for this operation
     * @return The l2 norm of the matrix
     */
    [[nodiscard]] float l2norm(utils::clWrapper &wrapper, cl::CommandQueue &queue) const;

    /**
     * @brief Sum the element of the matrix an return the result. This operation is always blocking,
     * and uses the default queue
     * @param wrapper  The wrapper to use for this operation
     * @return The l2 norm of the matrix
     */
    [[nodiscard]] float l2norm(utils::clWrapper &wrapper) const {
      return l2norm(wrapper, wrapper.getDefaultQueue());
    }

    [[nodiscard]] clFMatrix transpose(utils::clWrapper &wrapper, cl::CommandQueue &queue,
                                      bool blocking = false) const;
    [[nodiscard]] clFMatrix transpose(utils::clWrapper &wrapper, bool blocking = false) const {
      return transpose(wrapper, wrapper.getDefaultQueue(), blocking);
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
    void ipadd(const clFMatrix &other, utils::clWrapper &wrapper, cl::CommandQueue &queue,
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
    void ipadd(const clFMatrix &other, utils::clWrapper &wrapper, bool blocking = false) {
      ipadd(other, wrapper, wrapper.getDefaultQueue(), blocking);
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
    [[nodiscard]] clFMatrix add(const clFMatrix &other, utils::clWrapper &wrapper,
                                cl::CommandQueue &queue, bool blocking = false) const;

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
    [[nodiscard]] clFMatrix add(const clFMatrix &other, utils::clWrapper &wrapper,
                                bool blocking = false) const {
      return add(other, wrapper, wrapper.getDefaultQueue(), blocking);
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
    void ipsub(const clFMatrix &other, utils::clWrapper &wrapper, cl::CommandQueue &queue,
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
    void ipsub(const clFMatrix &other, utils::clWrapper &wrapper, bool blocking = false) {
      ipsub(other, wrapper, wrapper.getDefaultQueue(), blocking);
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
    [[nodiscard]] clFMatrix sub(const clFMatrix &other, utils::clWrapper &wrapper,
                                cl::CommandQueue &queue, bool blocking = false) const;

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
    [[nodiscard]] clFMatrix sub(const clFMatrix &other, utils::clWrapper &wrapper,
                                bool blocking = false) const {
      return sub(other, wrapper, wrapper.getDefaultQueue(), blocking);
    }

    void ipscale(float scale, utils::clWrapper &wrapper);

    [[nodiscard]] clFMatrix scale(float scale, utils::clWrapper &wrapper) const;
    clFMatrix iphadamard(const clFMatrix &other, utils::clWrapper &wrapper) const;

    clFMatrix hadamard(const clFMatrix &other, utils::clWrapper &wrapper) const;


    [[nodiscard]] static clFMatrix gemm(float alpha, bool transpose_a, const clFMatrix &A,
                                        bool transpose_b, const clFMatrix &B,
                                        utils::clWrapper &wrapper, cl::CommandQueue &queue,
                                        bool blocking = false);

    [[nodiscard]] static clFMatrix gemm(float alpha, bool transpose_a, const clFMatrix &A,
                                        bool transpose_b, const clFMatrix &B,
                                        utils::clWrapper &wrapper, bool blocking = false) {
      return gemm(alpha, transpose_a, A, transpose_b, B, wrapper, wrapper.getDefaultQueue(),
                  blocking);
    }

    [[nodiscard]] static clFMatrix gemm(float alpha, bool transpose_a, const clFMatrix &A,
                                        bool transpose_b, const clFMatrix &B, float beta,
                                        clFMatrix &C, utils::clWrapper &wrapper,
                                        cl::CommandQueue &queue, bool blocking = false);

    [[nodiscard]] static clFMatrix gemm(float alpha, bool transpose_a, const clFMatrix &A,
                                        bool transpose_b, const clFMatrix &B, float beta,
                                        clFMatrix &C, utils::clWrapper &wrapper,
                                        bool blocking = false) {
      return gemm(alpha, transpose_a, A, transpose_b, B, beta, C, wrapper,
                  wrapper.getDefaultQueue(), blocking);
    }

  private:
    cl::Buffer data;
    size_t rows = 0, cols = 0;
  };
}   // namespace math