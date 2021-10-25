#pragma once
#include <cstring>
#include <memory>

namespace math {

template <typename T> class Matrix {
public:
  // Create an empty matrix, with cols/rows of size 0
  // and no allocation
  // Allowing a matrix to be empty allows for easier copying
  // especially when storing them in an array
  Matrix() = default;

  Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {

    if (rows == 0 || cols == 0)
      return;

    data = new T[rows * cols];
  }

  ~Matrix() {
    if (data)
      delete[] data;
  }

  // Does not perform bound checking
  T &operator()(size_t i, size_t j) { return data[i * cols + j]; };

  Matrix(const Matrix &other) { *this = other; }

  Matrix(Matrix &&other) { *this = std::move(other); }

  Matrix &operator=(const Matrix &other) {
    // No reason to copy oneself
    if (this == &other)
      return *this;

    // We may be able to skip the reallacoation

    if (data && rows * cols != other.rows * other.cols) {
      delete data;
      data = nullptr;
    }

    rows = other.rows;
    cols = other.cols;

    // not need no copy an empty matrix
    if (other.data) {
      if (not data)
        data = new T(other.rows * other.cols);
      std::memcpy(data, other.data, sizeof(T) * rows * cols);
    }
    return *this;
  }

  Matrix &operator=(Matrix &&other) {
    // No reason to copy oneself
    if (this == &other)
      return *this;

    // We're overiding the old data
    if (data)
      delete data;

    data = other.data;
    rows = other.rows;
    cols = other.cols;

    other.data = nullptr;
    other.rows = 0;
    other.cols = 0;
    return *this;
  }

  T *begin() { return data; }

  T *end() { return data + (rows * cols); }

  const T *getData() const { return getData(); }

  T *getData() { return data; }

  size_t getRows() const { return rows; }

  size_t getCols() const { return cols; }

  T sumReduce() {

    if (not data)
      throw std::runtime_error("Cannot a null-sized matrix");

    T sum = 0;
    size_t stop = cols * rows;

    for (size_t i = 0; i < stop; i++) {
      sum += data[i];
    }

    return sum;
  }

  Matrix operator*(const Matrix other) {

    size_t b_rows = other.rows, b_cols = other.cols;

    if (cols != b_rows)
      throw std::invalid_argument("Matrix dimensions do not match");

    Matrix<T> c(rows, b_cols);

    T *raw_b = other.data;
    T *raw_c = c.data;

#ifdef USE_BLAS

    if constexpr (std::is_same_v<real, float>)
      static_assert(false, "blas not implemented");
    else if constexpr (std::is_same_v<real, double>)
      static_assert(false, "blas no implemented");
    else

#endif
      for (int i = 0; i < rows; i++) {
        for (int k = 0; k < cols; k++) {
          T a_ik = data[i * cols + k];
          for (int j = 0; j < b_cols; j++)
            raw_c[i * b_cols + j] += a_ik * raw_b[k * b_cols + j];
        }
      }

    return c;
  }

private:
  T *data = nullptr;
  size_t rows = 0, cols = 0;
};

using FloatMatrix = Matrix<float>;
using DoubleMatrix = Matrix<double>;

} // namespace math