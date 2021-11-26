#pragma once
extern "C" {
#include <cblas.h>
}
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>

//#define USE_BLAS


namespace math {

  template<typename T>
  class Matrix {
  public:
    enum class MatrixTranspose { NOTRANSPOSE, TRANSPOSE };

    // Create an empty matrix, with cols/rows of size 0
    // and no allocation
    // Allowing a matrix to be empty allows for easier copying
    // especially when storing them in an array
    Matrix() = default;

    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
      if (rows == 0 || cols == 0) { return; }

      data = std::make_unique<T[]>(rows * cols);
    }

    ~Matrix() = default;

    T *begin() { return data.get(); }

    const T *begin() const { return data.get(); }

    const T *cbegin() const { return data.get(); }

    T *end() { return data.get() + (rows * cols); }

    const T *end() const { return data.get() + (rows * cols); }

    const T *cend() const { return data.get() + (rows * cols); }

    const T *getData() const { return data.get(); }

    T *getData() { return data.get(); }

    [[nodiscard]] size_t getRows() const { return rows; }

    [[nodiscard]] size_t getCols() const { return cols; }

    /** @brief Returns element (i, j). Does not perform bound checking
     *
     * @param i
     * @param j
     * @return
     */
    T &operator()(size_t i, size_t j) { return data[i * cols + j]; };

    /** @brief Returns element (i, j). Does not perform bound checking
     *
     * @param i
     * @param j
     * @return
     */
    T const &operator()(size_t i, size_t j) const { return data[i * cols + j]; };

    Matrix(const Matrix &other) { *this = other; }

    Matrix(Matrix &&other) noexcept { *this = std::move(other); }

    Matrix &operator=(const Matrix &other) {
      // No reason to copy oneself
      if (this == &other) { return *this; }

      // NO need to copy an empty matrix
      if (other.data) {
        // If data is unallocated or different size, allocate new memory
        if (not data or data and rows * cols != other.rows * other.cols) {
          data = std::make_unique<T[]>(other.rows * other.cols);
        }
        rows = other.rows;
        cols = other.cols;
#ifdef USE_BLAS
        if constexpr (std::is_same_v<T, float>) {
          cblas_scopy(rows * cols, other.data.get(), 1, data.get(), 1);
        } else if constexpr (std::is_same_v<T, double>) {
          cblas_dcopy(rows * cols, other.data.get(), 1, data.get(), 1);
        }
#else
        std::memcpy(data.get(), other.getData(), sizeof(T) * rows * cols);
#endif
      }
      return *this;
    }

    Matrix &operator=(Matrix &&other) noexcept {
      // No reason to copy oneself
      if (this == &other) { return *this; }

      data = std::move(other.data);
      rows = other.rows;
      cols = other.cols;

      other.rows = 0;
      other.cols = 0;
      return *this;
    }

    [[nodiscard]] T sumReduce() const {
      if (not data) { throw std::runtime_error("Cannot sum-reduce a null-sized matrix"); }

      T sum = 0;
      const size_t stop = cols * rows;

      for (size_t i = 0; i < stop; i++) { sum += std::fabs(data[i]); }

      return sum;
    }

    [[nodiscard]] Matrix transpose() const {
      Matrix transposed(cols, rows);

      T *transposed_data = transposed.getData();
      for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) { transposed_data[j * rows + i] = data[i * cols + j]; }
      }

      return transposed;
    }

    Matrix &operator+=(const Matrix &other) {
      if (rows != other.rows or cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match");
      }

      const T *other_data = other.getData();
#ifdef USE_BLAS
      if constexpr (std::is_same_v<T, float>) {
        cblas_saxpy(rows * cols, 1.0f, other_data, 1, data.get(), 1);
      } else if constexpr (std::is_same_v<T, double>) {
        cblas_daxpy(rows * cols, 1.0, other_data, 1, data.get(), 1);
      }
#else
      for (size_t i = 0; i < rows * cols; i++) { data[i] += other_data[i]; }
#endif
      return *this;
    }

    [[nodiscard]] Matrix operator+(const Matrix &other) const {
      // To avoid copies, we need not to use the += operator
      // and directly perform the substraction in the result matrix
      // This is the reason behind this code duplicate
      // We could refactor this by creating an external method
      // but this raises issue with the const correctness
      if (rows != other.rows or cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match");
      }

      Matrix res(other);
      const T *other_data = other.getData();
      T *res_data = res.getData();

#ifdef USE_BLAS

      if constexpr (std::is_same_v<T, float>) {
        cblas_saxpy(rows * cols, 1.0f, data.get(), 1, res_data, 1);
      } else if constexpr (std::is_same_v<T, double>) {
        cblas_daxpy(rows * cols, 1.0, data.get(), 1, res_data, 1);
      }
#else
      for (size_t i = 0; i < rows * cols; i++) { res_data[i] = data[i] + other_data[i]; }
#endif
      return res;
    }

    Matrix &operator-=(const Matrix &other) {
      if (rows != other.rows or cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match");
      }

      Matrix res(rows, cols);
      const T *other_data = other.getData();

#ifdef USE_BLAS

      if constexpr (std::is_same_v<T, float>) {
        cblas_saxpy(rows * cols, -1.f, other_data, 1, data.get(), 1);
      } else if constexpr (std::is_same_v<T, double>) {
        cblas_daxpy(rows * cols, -1.0, other_data, 1, data.get(), 1);
      }
#else
      for (size_t i = 0; i < rows * cols; i++) { data[i] -= other_data[i]; }
#endif
      return *this;
    }

    [[nodiscard]] Matrix operator-(const Matrix &other) const {
      // To avoid copies, we need not use the -= operator
      // and directly perform the subtraction in the result matrix
      // This is the reason behind this code duplicate
      // We could refactor this by creating an external method
      // but this raises issue with the const correctness
      if (rows != other.rows or cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match");
      }

      Matrix res(*this);
      const T *other_data = other.getData();
      T *res_data = res.getData();

#ifdef USE_BLAS

      if constexpr (std::is_same_v<T, float>) {
        cblas_saxpy(rows * cols, -1.0f, other_data, 1, res_data, 1);
      } else if constexpr (std::is_same_v<T, double>) {
        cblas_daxpy(rows * cols, -1.0, other_data, 1, res_data, 1);
      }
#else
      for (size_t i = 0; i < rows * cols; i++) { res_data[i] = data[i] - other_data[i]; }
#endif
      return res;
    }

    [[nodiscard]] Matrix operator*(const Matrix &other) const {
      const size_t other_rows = other.rows, other_cols = other.cols;
      if (cols != other_rows) { throw std::invalid_argument("Matrix dimensions do not match"); }

      Matrix res(rows, other_cols);

      const T *raw_other = other.getData();
      T *raw_res = res.getData();

#ifdef USE_BLAS

      if constexpr (std::is_same_v<T, float>) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, other_cols, cols, 1.f,
                    data.get(), cols, raw_other, other_cols, 0.f, raw_res, other_cols);
      } else if constexpr (std::is_same_v<T, double>) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, other_cols, cols, 1.0,
                    data.get(), cols, raw_other, other_cols, 0.0, raw_res, other_cols);
      }
#else
      for (int i = 0; i < rows; i++) {
        for (int k = 0; k < cols; k++) {
          T a_ik = data[i * cols + k];
          for (int j = 0; j < other_cols; j++) {
            raw_res[i * other_cols + j] += a_ik * raw_other[k * other_cols + j];
          }
        }
      }
#endif
      return res;
    }

    [[nodiscard]] Matrix operator*(const T scale) const {
      Matrix res(*this);

      T *raw_res = res.getData();
      const T *raw_mat = data.get();

#ifdef USE_BLAS

      if constexpr (std::is_same_v<T, float>) {
        cblas_sscal(rows * cols, scale, raw_res, 1);
      } else if constexpr (std::is_same_v<T, double>) {
        cblas_dscal(rows * cols, scale, raw_res, 1);
      }
#else
      const size_t size{rows * cols};
      for (size_t i = 0; i < size; i++) { raw_res[i] *= scale; }
#endif
      return res;
    }

    Matrix &operator*=(const T scale) {
      T *raw_mat = data.get();

#ifdef USE_BLAS
      if constexpr (std::is_same_v<T, float>) {
        cblas_sscal(rows * cols, scale, raw_mat, 1);
      } else if constexpr (std::is_same_v<T, double>) {
        cblas_dscal(rows * cols, scale, raw_mat, 1);
      }
#else
      const size_t size{rows * cols};
      for (size_t i = 0; i < size; i++) { raw_mat[i] *= scale; }
#endif
      return *this;
    }

    void hadamardProd(const Matrix &other) const {
      if (rows != other.rows or cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match");
      }

      const T *raw_data_other = other.getData();
      T *raw_data = data.get();

      const size_t size{rows * cols};
      for (size_t i = 0; i < size; i++) { raw_data[i] *= raw_data_other[i]; }
    }

    [[nodiscard]] static Matrix matMatProdMatAdd(const Matrix &A, const Matrix &B,
                                                 const Matrix &C) {
      const size_t A_rows = A.rows, A_cols = A.cols, B_rows = B.rows, B_cols = B.cols,
                   C_rows = C.rows, C_cols = C.cols;

      if (A_cols != B_rows || A_rows != C_rows || B_cols != C_cols) {
        throw std::invalid_argument("Matrix dimensions do not match");
      }

      Matrix res(C);

#ifdef USE_BLAS
      if constexpr (std::is_same_v<T, float>) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A_rows, B_cols, A_cols, 1.f,
                    A.getData(), A_cols, B.getData(), B_cols, 1.f, res.getData(), C_cols);
      } else if constexpr (std::is_same_v<T, double>) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A_rows, B_cols, A_cols, 1.0,
                    A.getData(), A_cols, B.getData(), B_cols, 1.0, res.getData(), C_cols);
      }
#else
      res = A * B + C;
#endif
      return res;
    }

    [[nodiscard]] static Matrix matTransMatProd(const Matrix &A, const Matrix &B) {
      const size_t A_rows = A.rows, A_cols = A.cols, B_rows = B.rows, B_cols = B.cols;

      if (A_rows != B_rows) { throw std::invalid_argument("Matrix dimensions do not match"); }

      Matrix res(A_cols, B_cols);

#ifdef USE_BLAS
      if constexpr (std::is_same_v<T, float>) {
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, A_cols, B_cols, A_rows, 1.f,
                    A.getData(), A_cols, B.getData(), B_cols, 0.f, res.getData(), B_cols);
      } else if constexpr (std::is_same_v<T, double>) {
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, A_cols, B_cols, A_rows, 1.0,
                    A.getData(), A_cols, B.getData(), B_cols, 0.0, res.getData(), B_cols);
      }
#else
      res = A.transpose() * B;
#endif
      return res;
    }

    [[nodiscard]] static Matrix matMatTransProd(const Matrix &A, const Matrix &B) {
      const size_t A_rows = A.rows, A_cols = A.cols, B_rows = B.rows, B_cols = B.cols;

      if (A_cols != B_cols) { throw std::invalid_argument("Matrix dimensions do not match"); }

      Matrix res(A_rows, B_rows);

#ifdef USE_BLAS
      if constexpr (std::is_same_v<T, float>) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, A_rows, B_rows, A_cols, 1.f,
                    A.getData(), A_cols, B.getData(), B_cols, 0.f, res.getData(), res.getCols());
      } else if constexpr (std::is_same_v<T, double>) {
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, A_cols, B_cols, A_rows, 1.0,
                    A.getData(), A_cols, B.getData(), B_cols, 0.0, res.getData(), B_cols);
      }
#else
      res = A * B.transpose();
#endif
      return res;
    }

    [[nodiscard]] static Matrix mul(const bool transpose_a, const Matrix &A, const bool transpose_b,
                                    const Matrix &B, const T alpha = 1.0) {
      const size_t A_rows = A.rows, A_cols = A.cols, B_rows = B.rows, B_cols = B.cols;

      if ((transpose_a ? A_rows : A_cols) != (transpose_b ? B_cols : B_rows)) {
        throw std::invalid_argument("Matrix size do not match");
      }

      Matrix res((transpose_a ? A_cols : A_rows), (transpose_b ? B_rows : B_cols));

#ifdef USE_BLAS
      auto ta = transpose_a ? CblasTrans : CblasNoTrans;
      auto tb = transpose_b ? CblasTrans : CblasNoTrans;
      size_t m = (transpose_a ? A_cols : A_rows);
      size_t n = (transpose_b ? B_rows : B_cols);
      size_t k = (transpose_a ? A_rows : A_cols);

      if constexpr (std::is_same_v<T, float>) {
        cblas_sgemm(CblasRowMajor, ta, tb, m, n, k, alpha, A.getData(), A_cols, B.getData(), B_cols,
                    0.f, res.getData(), res.getCols());
      } else if constexpr (std::is_same_v<T, double>) {
        cblas_dgemm(CblasRowMajor, ta, tb, m, n, k, alpha, A.getData(), A_cols, B.getData(), B_cols,
                    0.1, res.getData(), res.getCols());
      }
#else
      if (!transpose_a && !transpose_b) {
        res = (A * alpha) * B;
      } else if (transpose_a && !transpose_b) {
        res = (A.transpose() * alpha) * B;
      } else if (!transpose_a && transpose_b) {
        res = (A * alpha) * B.transpose();
      } else {
        res = (A.transpose() * alpha) * B.transpose();
      }
#endif
      return res;
    }

  private:
    std::unique_ptr<T[]> data = nullptr;
    size_t rows = 0, cols = 0;
  };

  using FloatMatrix = Matrix<float>;
  using DoubleMatrix = Matrix<double>;

  template<typename T>
  std::ostream &operator<<(std::ostream &os, const Matrix<T> &m) {
    size_t j = 0;
    const size_t cols = m.getCols();

    for (T const &i : m) {
      os << i << " ";
      if (++j % cols == 0) { os << "\n"; }
    }
    return os;
  }

}   // namespace math