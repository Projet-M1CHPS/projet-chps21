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

    data = std::make_unique<T[]>(rows * cols);
  }

  ~Matrix() = default;

  T *begin() { return data.get(); }
  T *cbegin() const { return data.get(); }

  T *end() { return data.get() + (rows * cols); }
  T *cend() const { return data.get() + (rows * cols); }

  const T *getData() const { return data.get(); }

  T *getData() { return data.get(); }

  size_t getRows() const { return rows; }

  size_t getCols() const { return cols; }

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
      data = nullptr;
    }

    rows = other.rows;
    cols = other.cols;

    // not need no copy an empty matrix
    if (other.data) {
      if (not data)
        data = std::make_unique<T[]>(rows * cols);
      std::memcpy(data.get(), other.getData(), sizeof(T) * rows * cols);
    }
    return *this;
  }

  Matrix &operator=(Matrix &&other) {
    // No reason to copy oneself
    if (this == &other)
      return *this;

    data = std::move(other.data);
    rows = other.rows;
    cols = other.cols;

    other.rows = 0;
    other.cols = 0;
    return *this;
  }

  T sumReduce() const {

    if (not data)
      throw std::runtime_error("Cannot a null-sized matrix");

    T sum = 0;
    size_t stop = cols * rows;

    for (size_t i = 0; i < stop; i++) {
      sum += data[i];
    }

    return sum;
  }

  Matrix transpose() const {
    Matrix transposed(cols, rows);

    T* transposed_data = transposed.getData();
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        transposed_data[j * rows + i] = data[i * cols + j];
      }
    }

    return transposed;
  }

  Matrix &operator+=(const Matrix &other) {

    if (rows != other.rows or cols != other.cols)
      throw std::invalid_argument("Matrix dimensions do not match");

    const T *other_data = other.getData();
    T *res_data = getData();
#ifdef USE_BLAS

    if constexpr (std::is_same_v<real, float>)
      static_assert(false, "blas not implemented");
    else if constexpr (std::is_same_v<real, double>)
      static_assert(false, "blas not implemented");
    else
#endif
      for (size_t i = 0; i < rows * cols; i++) {
        res_data[i] = data[i] + other_data[i];
      }
    return *this;
  }

  Matrix operator+(const Matrix &other) const {

    if (rows != other.rows or cols != other.cols)
      throw std::invalid_argument("Matrix dimensions do not match");

    const T *other_data = other.getData();
    Matrix<T> res(rows, cols);

    T *res_data = res.getData();
#ifdef USE_BLAS

    if constexpr (std::is_same_v<real, float>)
      static_assert(false, "blas not implemented");
    else if constexpr (std::is_same_v<real, double>)
      static_assert(false, "blas not implemented");
    else
#endif
      for (size_t i = 0; i < rows * cols; i++) {
        res_data[i] = data[i] + other_data[i];
      }
    return res;
  }

  Matrix &operator-=(const Matrix &other) {
    Matrix res(rows, cols);
    const T *other_data = other.getData();
    T *res_data = getData();

#ifdef USE_BLAS

    if constexpr (std::is_same_v<real, float>)
      static_assert(false, "blas not implemented");
    else if constexpr (std::is_same_v<real, double>)
      static_assert(false, "blas not implemented");
    else
#endif
      for (size_t i = 0; i < rows * cols; i++) {
        res_data[i] = data[i] - other_data[i];
      }

    return *this;
  }

  Matrix operator-(const Matrix &other) const {
    if (rows != other.rows or cols != other.cols)
      throw std::invalid_argument("Matrix dimensions do not match");

    Matrix res(rows, cols);
    const T *other_data = other.getData();
    T *res_data = res.getData();

#ifdef USE_BLAS

    if constexpr (std::is_same_v<real, float>)
      static_assert(false, "blas not implemented");
    else if constexpr (std::is_same_v<real, double>)
      static_assert(false, "blas not implemented");
    else
#endif
      for (size_t i = 0; i < rows * cols; i++) {
        res_data[i] = data[i] - other_data[i];
      }

    return res;
  }

  Matrix operator*(const Matrix &other) const {

    size_t other_rows = other.rows, other_cols = other.cols;

    if (cols != other_rows)
      throw std::invalid_argument("Matrix dimensions do not match");

    Matrix res(rows, other_cols);

    const T *raw_other = other.getData();
    T *raw_res = res.getData();

#ifdef USE_BLAS

    if constexpr (std::is_same_v<real, float>)
      static_assert(false, "blas not implemented");
    else if constexpr (std::is_same_v<real, double>)
      static_assert(false, "blas not implemented");
    else

#endif
      for (int i = 0; i < rows; i++) {
        for (int k = 0; k < cols; k++) {
          T a_ik = data[i * cols + k];
          for (int j = 0; j < other_cols; j++)
            raw_res[i * other_cols + j] += a_ik * raw_other[k * other_cols + j];
        }
      }

    return res;
  }

private:
  std::unique_ptr<T[]> data = nullptr;
  size_t rows = 0, cols = 0;
};

using FloatMatrix = Matrix<float>;
using DoubleMatrix = Matrix<double>;

} // namespace math