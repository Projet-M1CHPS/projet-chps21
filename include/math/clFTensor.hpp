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

    cl::Buffer getBuffer() const { return data; }

    clFTensor sub(float factor, const clFTensor &other, cl::CommandQueue &queue,
                  bool blocking = false) const;

    static clFTensor batchedGemm(float alpha, bool transpose_a, const clFMatrix &A,
                                 bool transpose_b, const clFTensor &B, cl::CommandQueue &queue,
                                 bool blocking = false);

    static clFTensor batchedGemm(float alpha, bool transpose_a, const clFMatrix &A,
                                 bool transpose_b, const clFTensor &B, float beta,
                                 const clFMatrix &C, cl::CommandQueue &queue,
                                 bool blocking = false);


    static clFTensor batchedGemm(float alpha, bool transpose_a, const clFTensor &A,
                                 bool transpose_b, const clFTensor &B, cl::CommandQueue &queue,
                                 bool blocking = false);

    clFMatrix sumCollapse(cl::CommandQueue &queue, bool blocking = false) const;

    float sum() const {
      float sum = .0f;
      for (auto &item : getMatrices()) sum += item.sumReduce();
      return sum;
    }

    clFTensor &iphadamard(const clFTensor &other, cl::CommandQueue &queue, bool blocking = false);

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
