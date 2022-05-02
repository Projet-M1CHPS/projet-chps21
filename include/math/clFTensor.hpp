#pragma once
#include "clFMatrix.hpp"

namespace math {

  /**
   * @brief Represents a tensor of clFMatrix.
   * Each matrix is  of size x * y, the z dimension is the number of matrices.
   *
   * This class can be used for efficient batched matrix multiplication.
   */
  class clFTensor {
  public:
    friend std::ostream &operator<<(std::ostream &os, const clFTensor &t);

    clFTensor() : rows(0), cols(0), depth(0) {}
    clFTensor(size_t width, size_t height, size_t depth);

    clFTensor(const clFTensor &other, cl::CommandQueue &queue, bool blocking) {
      this->copy(other, queue, blocking);
    }

    clFTensor(const clFTensor &other, bool blocking)
        : clFTensor(other, utils::cl_wrapper.getDefaultQueue(), blocking) {}

    clFTensor(const clFTensor &other) = delete;
    clFTensor &operator=(const clFTensor &other) = delete;

    clFTensor(clFTensor &&other) = default;
    clFTensor &operator=(clFTensor &&other) = default;

    /**
     * @brief Performs a deep copy of the tensor.
     * @param queue The queue to use for the copy.
     * @param blocking If true, the operation will block until the copy is finished.
     * @return A new tensor with a copy of the data of this tensor
     */
    clFTensor &copy(const clFTensor &other, cl::CommandQueue &queue, bool blocking);

    /**
     * @brief Performs a shallow copy of this tensor, meaning that the new tensor shares the same
     * data. Any changes to the data of the new tensor will also affect the data of this tensor.
     * @return A shallow copy of this tensor
     */
    clFTensor shallowCopy() const;

    void fill(float elem, cl::CommandQueue &queue, bool blocking) {
      cl::Event event;
      queue.enqueueFillBuffer(data, elem, getOffsetInBytes(), sizeInBytes(), nullptr, &event);
      if (blocking) event.wait();
    }

    /**
     * @brief Divides the tensor into multiple chunks
     * @param ndiv
     * @return
     */
    std::vector<clFTensor> slice(size_t ndiv) const;

    /**
     * @brief Slice this tensor and return the tensor(rows, cols, begin:end)
     * @param begin The first matrix of the slice
     * @param end The last matrix of the slice, not included ([begin, end[)
     * @return A slice of this tensor
     */
    clFTensor slice(size_t begin, size_t end) const;

    /**
     * @brief Return the offset in floats of the matrix at the given index.
     * @param matrix_index
     * @return
     */
    size_t getOffsetOf(size_t matrix_index) const { return (matrix_index + offset) * rows * cols; }

    /**
     * @brief Returns the offset in bytes of the matrix at the given index.
     * @param index
     * @return
     */
    size_t getOffsetOfInBytes(size_t matrix_index) const {
      return getOffsetOf(matrix_index) * sizeof(float);
    }

    /**
     * @brief Returns the offset in matrix of this tensor.
     * @return
     */
    size_t getOffset() const { return offset; }

    /**
     * @brief Returns the offset in bytes of this tensor.
     * @return
     */
    size_t getOffsetInBytes() const { return offset * rows * cols * sizeof(float); }

    /**
     * @brief Returns the offsets in floats of this tensor.
     * @return
     */
    size_t getOffsetInFloats() const { return offset * rows * cols; }

    /**
     * @brief Returns a clFTensor wheres matrices are flattened (x * y, 1, z)
     * Beware that this operation does not copy the matrices, and any change to the matrices will be
     * reflected in the original clFTensor.
     *
     * @param tensor
     * @return
     */
    clFTensor flatten() const;

    /**
     * @brief Returns the submatrix at the given index.
     * Note that the returned matrix is a view of the internal matrix, and any change made to it
     * will be reflected in the tensor
     *
     * Submatrices created this way remains valid even if the tensor is destroyed, since openCl
     * keeps track of buffers and subbuffers.
     *
     * @param z The index of the submatrix.
     * @return A submatrix inside the tensor, throws on error
     */
    [[deprecated("Use operator[]. Will be removed in future versions")]] clFMatrix
    getMatrix(size_t z);

    /**
     * @brief Returns the submatrix at the given index.
     * Note that the returned matrix is a view of the internal matrix, and any change made to it
     * will be reflected in the tensor
     *
     * Submatrices created this way remains valid even if the tensor is destroyed, since openCl
     * keeps track of buffers and subbuffers.
     *
     * @param z The index of the submatrix.
     * @return A submatrix inside the tensor, throws on error
     */
    [[deprecated("Use operator[]. Will be removed in future versions")]] clFMatrix
    getMatrix(size_t z) const;

    clFMatrix operator[](size_t z) {
      if (z > depth) { throw std::out_of_range("clFTensor::getMatrix: z index out of range"); }

      return {data, rows, cols, getOffsetOf(z)};
    }

    clFMatrix operator[](size_t z) const {
      if (z > depth) { throw std::out_of_range("clFTensor::getMatrix: z index out of range"); }

      return {data, rows, cols, getOffsetOf(z)};
    }

    /**
     * @brief Returns an array of submatrices.
     *
     * * Submatrices created this way remains valid even if the tensor is destroyed, since openCl
     * keeps track of buffers and subbuffers.
     * @return An array of submatrices inside the tensor
     */
    [[nodiscard]] std::vector<clFMatrix> getMatrices();
    [[nodiscard]] std::vector<clFMatrix> getMatrices() const;

    /**
     * @brief Get the x dimension of the tensor, corresponding to the number of columns in each
     * matrix
     * @return
     */
    size_t getRows() const { return rows; }

    /**
     * @brief Get the y dimension of the tensor, corresponding to the number of rows in each matrix
     * @return
     */
    size_t getCols() const { return cols; }

    /**
     * @brief Get the z dimension of the tensor, corresponding to the number of matrices
     * @return
     */
    size_t getDepth() const { return depth; }

    size_t size() const { return rows * cols * depth; }
    size_t sizeInBytes() const { return size() * sizeof(float); }

    /**
     * @brief Returns the opencl buffer associated with the tensor
     * @return
     */
    cl::Buffer getBuffer() const { return data; }

    /**
     * @brief Reshapes the tensor to the given dimensions
     * @param new_rows
     * @param new_cols
     * @param new_depth
     */
    void reshape(size_t new_rows, size_t new_cols, size_t new_depth);

    /**
     * @brief C = A - alpha * B
     * @param alpha
     * @param other
     * @param queue
     * @param blocking If true, blocks until the operation is complete
     * @return
     */
    clFTensor sub(float alpha, const clFTensor &other, cl::CommandQueue &queue,
                  bool blocking = false) const;

    /**
     * @brief C = C + alpha * B, where B is a tensor
     * @param alpha
     * @param B
     * @param queue
     * @param blocking If true, blocks until the operation is complete
     */
    void ipadd(float alpha, const clFTensor &B, cl::CommandQueue &queue, bool blocking = false);

    /**
     * @brief C = alpha * A * B, where A is a matrix and B is a tensor
     * @param alpha
     * @param transpose_a If true transposes A
     * @param A
     * @param transpose_b If true transposes B
     * @param B
     * @param queue
     * @param blocking If true, blocks until the operation is complete
     * @return
     */
    static clFTensor batchedGemm(float alpha, bool transpose_a, const clFMatrix &A,
                                 bool transpose_b, const clFTensor &B, cl::CommandQueue &queue,
                                 bool blocking = false);

    /**
     * @brief R = alpha * A * B + beta * C, where A is a matrix, B a tensor, and C a matrix
     * @param alpha
     * @param transpose_a If true transposes A
     * @param A
     * @param transpose_b If true transposes B
     * @param B
     * @param beta
     * @param C
     * @param queue
     * @param blocking If true, blocks until the operation is complete
     * @return
     */
    static clFTensor batchedGemm(float alpha, bool transpose_a, const clFMatrix &A,
                                 bool transpose_b, const clFTensor &B, float beta,
                                 const clFMatrix &C, cl::CommandQueue &queue,
                                 bool blocking = false);

    /**
     * @brief C = alpha * A * B, where A and B are tensors
     * @param alpha
     * @param transpose_a If true transposes A
     * @param A
     * @param transpose_b If true transposes B
     * @param B
     * @param queue
     * @param blocking If true, blocks until the operation is complete
     * @return
     */
    static clFTensor batchedGemm(float alpha, bool transpose_a, const clFTensor &A,
                                 bool transpose_b, const clFTensor &B, cl::CommandQueue &queue,
                                 bool blocking = false);

    /**
     * @brief Sums the tensors along the z-axis, returning a single matrix
     * @param queue
     * @param blocking If true, blocks until the operation is complete
     * @return
     */
    clFMatrix sumCollapse(cl::CommandQueue &queue, bool blocking = false) const;

    /**
     * @brief Multiply two tensors element by element
     * @param other
     * @param queue
     * @param blocking If true, blocks until the operation is complete
     * @return
     */
    clFTensor &iphadamard(const clFTensor &other, cl::CommandQueue &queue, bool blocking = false);

    /**
     * @brief Inplace Scale every element of the tensor by a factor.
     *
     * @param factor The factor to scale the matrix with
     * @param queue The queue to use for this operation
     * @param blocking True if the operation is blocking, false otherwise
     */
    void ipscale(float factor, cl::CommandQueue &queue, bool blocking = false);

  private:
    cl::Buffer data;
    size_t rows = 0, cols = 0, depth = 0;
    // Offset of the first element in the tensor, in element (number of matrix to skip)
    size_t offset = 0;

    // If true, this tensor is a view of another tensor
    // and should never be resized
    bool is_view = false;
  };
}   // namespace math
