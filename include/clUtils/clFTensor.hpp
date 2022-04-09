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
    clFTensor() : x_dim(0), y_dim(0), z_dim(0) {}
    clFTensor(size_t x, size_t y, size_t z);


    clFTensor(const clFTensor &other, cl::CommandQueue &queue, bool blocking = true) {
      this->copy(other, queue, blocking);
    }
    clFTensor(const clFTensor &other, bool blocking = true)
        : clFTensor(other, utils::cl_wrapper.getDefaultQueue(), blocking) {}

    clFTensor(clFTensor &&other) noexcept { *this = std::move(other); }
    clFTensor &operator=(clFTensor &&other) noexcept = default;

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
    clFTensor shallowCopy() const {
      clFTensor copy;
      copy.x_dim = x_dim;
      copy.y_dim = y_dim;
      copy.z_dim = z_dim;
      copy.data = data;
      copy.offset = offset;
      return copy;
    }

    /**
     * @brief Split the tensor in @ndiv parts. The resulting tensor are only a shallow copy.
     * If the size of the tensor is not divisible by @ndiv, the last part will be smaller.
     * @param ndiv The number of parts to split the tensor in.
     * @return A vector containing the split tensors.
     */
    std::vector<clFTensor> shallowSplit(size_t ndiv) const;

    /**
     * @brief Returns a clFTensor wheres matrices are flattened (x * y, 1)
     * Beware that this operation does not copy the matrices, and any change to the matrices will be
     * reflected in the original clFTensor.
     *
     * @param tensor
     * @return
     */
    clFTensor flatten() const {
      clFTensor res;
      res.data = data;
      res.x_dim = x_dim * y_dim;
      res.y_dim = y_dim > 0 ? 1 : 0;
      res.z_dim = z_dim;
      res.offset = offset;
      return res;
    }

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

    clFMatrix meanSumCollapse(cl::CommandQueue &queue, cl::Event &event,
                              bool blocking = false) const;

    clFTensor &iphadamard(const clFTensor &other, cl::CommandQueue &queue, bool blocking = false);

    size_t offsetOf(size_t index);
    size_t offsetOfInBytes(size_t index);
    size_t getOffset();

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
    clFMatrix getMatrix(size_t z);

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
    clFMatrix getMatrix(size_t z) const;

    /**
     * @brief Returns an array of submatrices.
     *
     * * Submatrices created this way remains valid even if the tensor is destroyed, since openCl
     * keeps track of buffers and subbuffers.
     * @return An array of submatrices inside the tensor
     */
    std::vector<clFMatrix> getMatrices();

    /**
     * @brief Get the x dimension of the tensor, corresponding to the number of columns in each
     * matrix
     * @return
     */
    size_t getX() const { return x_dim; }

    /**
     * @brief Get the y dimension of the tensor, corresponding to the number of rows in each matrix
     * @return
     */
    size_t getY() const { return y_dim; }

    /**
     * @brief Get the z dimension of the tensor, corresponding to the number of matrices
     * @return
     */
    size_t getZ() const { return z_dim; }

    cl::Buffer getBuffer() const { return data; }

  private:
    cl::Buffer data;
    size_t x_dim = 0;
    size_t y_dim = 0;
    size_t z_dim = 0;
    size_t offset = 0;
  };

}   // namespace math
