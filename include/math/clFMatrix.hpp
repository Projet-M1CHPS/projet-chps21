#pragma once
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS 1
#include "Matrix.hpp"
#include "clWrapper.hpp"
#include <CL/opencl.hpp>
#include <clblast.h>
#include <utility>

namespace math {

  /**
   * @brief A float matrix stored in a cl::Buffer. Provides wrappers for most clblast operations.
   *
   *
   * As opposed to math::FloatMatrix, multiple matrices can share the same cl::Buffer, being views
   * of the same data. The memory is freed when the last view is destroyed.
   *
   * This class uses the default platform/context, and should therefore only be used after
   * clWrapper::initOpenCL() has been called..
   */
  class clFMatrix {
  public:
    friend std::ostream &operator<<(std::ostream &os, const clFMatrix &m);

    /**
     * @brief Creates an empty matrix
     */
    clFMatrix() = default;

    /**
     * @brief Convert a FloatMatrix to a clFMatrix, using the default queue, and block until the
     * copy is done.
     * @param other The matrix to convert
     * @return The matrix itself after the copy
     */
    clFMatrix &operator=(const FloatMatrix &other);

    clFMatrix(clFMatrix &&) = default;
    clFMatrix &operator=(clFMatrix &&) = default;

    /**
     * @brief Allocates a new matrix of the given size. Data is left uninitialized.
     * @param rows Numbers of rows of the new matrix
     * @param cols Numbers of columns of the new matrix
     */
    clFMatrix(size_t rows, size_t cols);

    /**
     * @brief Allocates a new matrix from a raw float array, and copy the content of the host array
     * to it
     * @param source The source float array to copy from
     * @param rows The numbers of rows of the new matrix
     * @param cols The numbers of cols of the new matrix
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
     * @param blocking True if the operation is blocking, false otherwise
     * Note that if the operation is non-blocking, the user is responsible for ensuring that the
     * matrix remains valid until the operation is finished
     */
    clFMatrix(const float *source, size_t rows, size_t cols, bool blocking)
        : clFMatrix(source, rows, cols, utils::cl_wrapper.getDefaultQueue(), blocking) {}

    /**
     * @brief Copies a FloatMatrix to the device
     * @param matrix the matrix to copy
     * @param queue The queue to be used for the copy operation
     * @param blocking True if the operation is blocking, false otherwise
     * Note that if the operation is non-blocking, the user is responsible for ensuring that the
     * matrix remains valid until the operation is finished
     */
    clFMatrix(const math::FloatMatrix &matrix, cl::CommandQueue &queue, bool blocking);

    /**
     * @brief Copies a FloatMatrix to the device, using the default queue
     * @param matrix The matrix to copy
     * @param blocking  True if the operation is blocking, false otherwise
     */
    explicit clFMatrix(const math::FloatMatrix &matrix, bool blocking = true)
        : clFMatrix(matrix, utils::cl_wrapper.getDefaultQueue(), blocking) {}


    clFMatrix &copy(const clFMatrix &other, cl::CommandQueue &queue, bool blocking);
    clFMatrix &copy(const clFMatrix &other, bool blocking) {
      return copy(other, utils::cl_wrapper.getDefaultQueue(), blocking);
    }

    /**
     * @brief Creates a new matrix from an existing OpenCL buffer
     * The buffer is not copied, the new matrix is just a wrapper around the existing buffer
     *
     * @param subbuffer The OpenCL buffer to be used for the matrix
     * @param rows The number of rows of the matrix
     * @param cols The number of cols of the matrix
     * @param offset The offset in elements (float), starting from the beginning of the buffer
     */
    clFMatrix(cl::Buffer &subbuffer, size_t width, size_t height, size_t offset);

    /**
     * @brief Creates a new matrix from an existing OpenCL buffer
     * The buffer is not copied, the new matrix is just a wrapper around the existing buffer
     *
     * @param subbuffer The OpenCL buffer to be used for the matrix
     * @param rows The number of rows of the matrix
     * @param cols The number of cols of the matrix
     * @param offset The offset in elements (float), starting from the beginning of the buffer
     */
    clFMatrix(const cl::Buffer &subbuffer, size_t width, size_t height, size_t offset);

    /**
     * @brief Reinterpret the matrix as a flat vector, without copying the data
     * Beware that the matrix is not copied, so any modification to the matrix will be reflected in
     * the original matrix
     * @return A flat vector
     */
    clFMatrix flatten() const;

    cl::Buffer &getBuffer() { return data; }
    [[nodiscard]] const cl::Buffer &getBuffer() const { return data; }

    [[nodiscard]] size_t getRows() const { return rows; }
    [[nodiscard]] size_t getCols() const { return cols; }
    [[nodiscard]] size_t getOffset() const { return offset; }
    [[nodiscard]] size_t getOffsetInBytes() const { return offset * sizeof(float); }
    [[nodiscard]] size_t size() const { return rows * cols; }
    [[nodiscard]] size_t sizeInBytes() const { return size() * sizeof(float); }

    /**
     * @brief Copies a matrix on the host to the device, replacing the current matrix
     * @param matrix The matrix to copy
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
     * @param blocking True if the operation is blocking, false otherwise
     * Note that if this operation is non-blocking, the user is responsible for ensuring that the
     * operation is finished before using the matrix
     * @return The new matrix
     */
    [[nodiscard]] FloatMatrix toFloatMatrix(cl::CommandQueue &queue, bool blocking = true) const;

    /**
     * @brief Copies the matrix on the device to a matrix on the host using the default queue
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
     * @param queue The queue to use for this operation
     * @return The sum of the elements of the matrix
     */
    [[nodiscard]] float sumReduce(cl::CommandQueue &queue) const;

    /**
     * @brief Fill the matrix with a constant value
     */
    clFMatrix &fill(float value, cl::CommandQueue &queue, bool blocking = true);

    /**
     * @brief Sum the element of the matrix an return the result. This operation is always blocking,
     * and uses the default queue
     * @return The sum of the elements of the matrix
     */
    [[nodiscard]] float sumReduce() const { return sumReduce(utils::cl_wrapper.getDefaultQueue()); }

    /**
     * @brief Sum the element of the matrix an return the result. This operation is always blocking
     * @param queue The queue to use for this operation
     * @return The l2 norm of the matrix
     */
    [[nodiscard]] float l2norm(cl::CommandQueue &queue) const;

    /**
     * @brief Sum the element of the matrix an return the result. This operation is always blocking,
     * and uses the default queue
     * @return The l2 norm of the matrix
     */
    [[nodiscard]] float l2norm() const { return l2norm(utils::cl_wrapper.getDefaultQueue()); }


    /**
     * @brief Return the index of the maximum element of the matrix
     * Note that this operation is blocking
     * @param queue The queue to use for this operation
     * @return
     */
    size_t imax(cl::CommandQueue &queue) const;

    size_t imax() const { return imax(utils::cl_wrapper.getDefaultQueue()); }

    [[nodiscard]] clFMatrix transpose(cl::CommandQueue &queue, bool blocking = false) const;

    [[nodiscard]] clFMatrix transpose(bool blocking = false) const {
      return transpose(utils::cl_wrapper.getDefaultQueue(), blocking);
    }

    /**
     * @brief inplace addition of two matrices. by default, this operation is non-blocking.
     * a = a + factor * b
     *
     * @param factor the factor to multiply the operand with
     * @param other the other matrix to add
     * @param queue the queue to use for this operation
     * @param blocking true if the operation is blocking, false otherwise
     * note that if this operation is non-blocking, the user is responsible for ensuring that the
     * operation is finished before using the matrix
     * @return this matrix after the addition
     */
    void ipadd(float factor, const clFMatrix &other, cl::CommandQueue &queue,
               bool blocking = false);

    /**
     * @brief Inplace addition of two matrices. By default, this operation is non-blocking, and uses
     the default queue.
     * A = A + factor * B
     *
     * @param factor The factor to multiply the operand with
     * @param other The other matrix to add
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
     * C = A + factor * B
     * @param factor The factor to multiply the operand with
     * @param other The other matrix to add
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
     * C = A + factor * B
     *
     * @param factor The factor to multiply the operand with
     * @param other The other matrix to add
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
     * A = A - factor * B
     *
     * @param factor The factor to multiply the operand with
     * @param other The other matrix to subtract
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
     * A = A - factor * B
     *
     * @param factor The factor to multiply the operand with
     * @param other The other matrix to subtract
     * @param blocking True if the operation is blocking, false otherwise
     * Note that if this operation is non-blocking, the user is responsible for ensuring that the
     * operation is finished before using the matrix
     */
    void ipsub(float factor, const clFMatrix &other, bool blocking = false) {
      ipsub(factor, other, utils::cl_wrapper.getDefaultQueue(), blocking);
    }

    /**
     * @brief Subtracts two matrices. By default, this operation is non-blocking
     * C = A - factor * B
     *
     * @param factor The factor to multiply the operand with
     * @param other The other matrix to subtract
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
     * C = A - factor * B
     *
     * @param factor The factor to multiply the operand with
     * @param other The other matrix to subtract
     * @param blocking True if the operation is blocking, false otherwise
     * Note that if this operation is non-blocking, the user is responsible for ensuring that the
     * operation is finished before using the matrix
     * @return this - other
     */
    [[nodiscard]] clFMatrix sub(float factor, const clFMatrix &other, bool blocking = false) const {
      return sub(factor, other, utils::cl_wrapper.getDefaultQueue(), blocking);
    }

    /**
     * @brief Inplace Scale every element of the matrix by a factor.
     *
     * @param scale The factor to scale the matrix with
     * @param queue The queue to use for this operation
     * @param blocking True if the operation is blocking, false otherwise
     */
    void ipscale(float scale, cl::CommandQueue &queue, bool blocking = false);

    /**
     * @brief Inplace Scale every element of the matrix by a factor, using the default queue.
     *
     * @param scale The factor to scale the matrix with
     * @param blocking True if the operation is blocking, false otherwise
     */
    void ipscale(float scale, bool blocking = false) {
      ipscale(scale, utils::cl_wrapper.getDefaultQueue(), blocking);
    }

    /**
     * @brief Scales every element of the matrix by a factor.
     *
     * @param scale The factor to scale the matrix with
     * @param queue The queue to use for this operation
     * @param blocking True if the operation is blocking, false otherwise
     * @return this * scale
     */
    [[nodiscard]] clFMatrix scale(float scale, cl::CommandQueue &queue,
                                  bool blocking = false) const;

    /**
     * @brief Scales every element of the matrix by a factor, using the default queue.
     *
     * @param scale The factor to scale the matrix with
     * @param blocking True if the operation is blocking, false otherwise
     * @return this * scale
     */
    [[nodiscard]] clFMatrix scale(float s, bool blocking = false) const {
      return scale(s, utils::cl_wrapper.getDefaultQueue(), blocking);
    }

    /**
     * @brief Inplace element-wise multiplication of two matrices. By default, this operation is
     * non-blocking and uses the default queue
     * A = A * B
     *
     * @param other The other matrix to multiply
     * @param queue The queue to use for this operation
     * @param blocking True if the operation is blocking, false otherwise
     * Note that if this operation is non-blocking, the user is responsible for ensuring that the
     * operation is finished before using the matrix
     */
    void iphadamard(const clFMatrix &other, cl::CommandQueue &queue, bool blocking = false) const;

    /**
     * @brief Inplace element-wise multiplication of two matrices. By default, this operation is
     * non-blocking and uses the default queue
     * A = A * B
     *
     * @param other The other matrix to multiply
     * @param blocking True if the operation is blocking, false otherwise
     * Note that if this operation is non-blocking, the user is responsible for ensuring that the
     * operation is finished before using the matrix
     */
    void iphadamard(const clFMatrix &other, bool blocking = false) const {
      iphadamard(other, utils::cl_wrapper.getDefaultQueue(), blocking);
    }

    /**
     * @brief Element-wise multiplication of two matrices. By default, this operation is
     * non-blocking and uses the default queue
     * C = A * B
     *
     * @param other The other matrix to multiply
     * @param queue The queue to use for this operation
     * @param blocking True if the operation is blocking, false otherwise
     * Note that if this operation is non-blocking, the user is responsible for ensuring that the
     * operation is finished before using the matrix
     * @return this * other
     */
    [[nodiscard]] clFMatrix hadamard(const clFMatrix &other, cl::CommandQueue &queue,
                                     bool blocking = false) const;

    /** @brief Element-wise multiplication of two matrices. By default, this operation is
     * non-blocking and uses the default queue
     * C = A * B
     *
     * @param other The other matrix to multiply
     * @param blocking True if the operation is blocking, false otherwise
     * Note that if this operation is non-blocking, the user is responsible for ensuring that the
     * operation is finished before using the matrix
     * @return this * other
     */
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

    // Offset in elements from the beginning of the buffer
    // This may be used when the matrix is part of a larger structure
    // For clblast calls, the offset must be specified in elements.
    // For OpenCL calls, the offset must be specified in bytes
    // This is why you may see this value multiplied by sizeof(float)
    size_t offset = 0;
  };
}   // namespace math