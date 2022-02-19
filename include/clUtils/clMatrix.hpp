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
    clMatrix &operator=(const clMatrix &) = delete;

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

    FloatMatrix toFloatMatrix(utils::clWrapper &wrapper) const {
      FloatMatrix matrix(rows, cols);
      wrapper.getDefaultQueue().enqueueReadBuffer(data, CL_TRUE, 0, rows * cols * sizeof(float),
                                                  (void *) matrix.getData());
      return matrix;
    }

    cl::Buffer &getBuffer() { return data; }
    const cl::Buffer &getBuffer() const { return data; }

    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }


    [[nodiscard]] float sumReduce() const {
      if (not data) { throw std::runtime_error("Cannot sum-reduce a null-sized matrix"); }

      float sum = 0;
      const size_t stop = cols * rows;

      for (size_t i = 0; i < stop; i++) { sum += data[i]; }

      return sum;
    }

    [[nodiscard]] float l2norm() const {
      if (not data) { throw std::runtime_error("Cannot sum-reduce a null-sized matrix"); }

      float sum = 0;
      const size_t stop = cols * rows;

      for (size_t i = 0; i < stop; i++) { sum += data[i] * data[i]; }

      return std::sqrt(sum);
    }


    [[nodiscard]] clMatrix transpose() const {
      clMatrix transposed(cols, rows);

      for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) { transposed_data[j * rows + i] = data[i * cols + j]; }
      }

      return transposed;
    }

    clMatrix &ipadd(const clMatrix &other) {
      if (rows != other.rows or cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match");
      }

      cblas_saxpy(rows * cols, 1.0f, other_data, 1, data.get(), 1);
      return *this;
    }

    [[nodiscard]] clMatrix add(const clMatrix &other) const {
      // To avoid copies, we need not to use the += operator
      // and directly perform the substraction in the result matrix
      // This is the reason behind this code duplicate
      // We could refactor this by creating an external method
      // but this raises issue with the const correctness
      if (rows != other.rows or cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match");
      }

      clMatrix res(other);
      const T *other_data = other.getData();
      T *res_data = res.getData();

      cblas_saxpy(rows * cols, 1.0f, data.get(), 1, res_data, 1);
      return res;
    }

    clMatrix &ipsub(const clMatrix &other) {
      if (rows != other.rows or cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match");
      }

      Matrix res(rows, cols);
      cblas_saxpy(rows * cols, -1.f, other_data, 1, data.get(), 1);
      return *this;
    }

    [[nodiscard]] clMatrix sub(const clMatrix &other) const {
      // To avoid copies, we need not use the -= operator
      // and directly perform the subtraction in the result matrix
      // This is the reason behind this code duplicate
      // We could refactor this by creating an external method
      // but this raises issue with the const correctness
      if (rows != other.rows or cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match");
      }

      Matrix res(*this);

      cblas_saxpy(rows * cols, -1.0f, other.getBuffer(), 1, res_data, 1);
      return res;
    }

    [[nodiscard]] clMatrix ipmul(const clMatrix &other) const {
      const size_t other_rows = other.rows, other_cols = other.cols;
      if (cols != other_rows) { throw std::invalid_argument("Matrix dimensions do not match"); }

      clMatrix res(rows, other_cols);

      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, other_cols, cols, 1.f,
                  data.get(), cols, raw_other, other_cols, 0.f, raw_res, other_cols);
      return res;
    }

    [[nodiscard]] clMatrix scale(const float scale) const {
      clMatrix res(*this);

      cblas_sscal(rows * cols, scale, buffer, 1);
      return res;
    }

    clMatrix &ipscale(const float scale) {
      cblas_sscal(rows * cols, scale, raw_mat, 1);
    }

    void hadamardProd(const Matrix &other) const {
      if (rows != other.rows or cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match");
      }

      const size_t size{rows * cols};
      for (size_t i = 0; i < size; i++) { raw_data[i] *= raw_data_other[i]; }
    }

    [[nodiscard]] static clMatrix matMatProdMatAdd(const Matrix &A, const Matrix &B,
                                                   const Matrix &C) {
      const size_t A_rows = A.rows, A_cols = A.cols, B_rows = B.rows, B_cols = B.cols,
                   C_rows = C.rows, C_cols = C.cols;

      if (A_cols != B_rows || A_rows != C_rows || B_cols != C_cols) {
        throw std::invalid_argument("Matrix dimensions do not match");
      }

      Matrix res(C);

      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A_rows, B_cols, A_cols, 1.f,
                  A.getData(), A_cols, B.getData(), B_cols, 1.f, res.getData(), C_cols);
      return res;
    }

    [[nodiscard]] static clMatrix matTransMatProd(const clMatrix &A, const clMatrix &B) {
      const size_t A_rows = A.rows, A_cols = A.cols, B_rows = B.rows, B_cols = B.cols;

      if (A_rows != B_rows) { throw std::invalid_argument("Matrix dimensions do not match"); }

      clMatrix res(A_cols, B_cols);

      cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, A_cols, B_cols, A_rows, 1.f, A.getData(),
                  A_cols, B.getData(), B_cols, 0.f, res.getData(), B_cols);
      return res;
    }

    [[nodiscard]] static clMatrix matMatTransProd(const clMatrix &A, const clMatrix &B) {
      const size_t A_rows = A.rows, A_cols = A.cols, B_rows = B.rows, B_cols = B.cols;

      if (A_cols != B_cols) { throw std::invalid_argument("Matrix dimensions do not match"); }

      clMatrix res(A_rows, B_rows);
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, A_rows, B_rows, A_cols, 1.f, A.getData(),
                  A_cols, B.getData(), B_cols, 0.f, res.getData(), res.getCols());
      return res;
    }

    [[nodiscard]] static clMatrix mul(const bool transpose_a, const clMatrix &A,
                                      const bool transpose_b, const clMatrix &B,
                                      const float alpha = 1.0) {
      const size_t A_rows = A.rows, A_cols = A.cols, B_rows = B.rows, B_cols = B.cols;

      if ((transpose_a ? A_rows : A_cols) != (transpose_b ? B_cols : B_rows)) {
        throw std::invalid_argument("Matrix size do not match");
      }

      clMatrix res((transpose_a ? A_cols : A_rows), (transpose_b ? B_rows : B_cols));

      auto ta = transpose_a ? CblasTrans : CblasNoTrans;
      auto tb = transpose_b ? CblasTrans : CblasNoTrans;
      size_t m = (transpose_a ? A_cols : A_rows);
      size_t n = (transpose_b ? B_rows : B_cols);
      size_t k = (transpose_a ? A_rows : A_cols);

      cblas_sgemm(CblasRowMajor, ta, tb, m, n, k, alpha, A.getBuffer(), A_cols, B.getBuffer(),
                  B_cols, 0.f, res.getBuffer(), res.getCols());
      return res;
    }

  private:
    cl::Buffer data;
    size_t rows = 0, cols = 0;
  };
}   // namespace math