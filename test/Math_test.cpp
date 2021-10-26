#include "Matrix.hpp"
#include "Utils.hpp"
#include <gtest/gtest.h>

using namespace math;

TEST(MatrixTest, CanCreateMatrix) {

  Matrix<float> m(10, 11);

  ASSERT_TRUE(m.getData());
  ASSERT_EQ(10, m.getRows());
  ASSERT_EQ(11, m.getCols());

  // Should be able to create an empty matrix
  const Matrix<float> n;

  ASSERT_EQ(nullptr, n.getData());
  ASSERT_EQ(0, n.getRows());
  ASSERT_EQ(0, n.getCols());
}

TEST(MatrixTest, CanCopyMatrix) {
  Matrix<float> m(2, 2), n(2, 2);

  m(0, 0) = 1;
  m(0, 1) = 1;
  m(1, 0) = 1;
  m(1, 1) = 1;

  n = m;

  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++)
      ASSERT_EQ(1, n(i, j));
}

TEST(MatrixTest, CanMoveCopy) {
  Matrix<float> m(2, 2);

  m(0, 0) = 1;
  m(0, 1) = 1;
  m(1, 0) = 1;
  m(1, 1) = 1;

  float *raw_ptr = m.getData();

  Matrix<float> n = std::move(m);
  // The move_copy should not reallocate an array
  ASSERT_EQ(raw_ptr, n.getData());
  ASSERT_EQ(2, n.getCols());
  ASSERT_EQ(2, n.getRows());

  // The src matrix should be reset to an empty matrix
  ASSERT_EQ(nullptr, m.getData());
  ASSERT_EQ(0, m.getRows());
  ASSERT_EQ(0, m.getCols());

  // should be able to copy an empty matrix
  Matrix<float> m1;

  n = m1;
  // The src matrix should be reset to an empty matrix
  ASSERT_EQ(nullptr, n.getData());
  ASSERT_EQ(0, n.getRows());
  ASSERT_EQ(0, n.getCols());
}

TEST(MatrixTest, CanIterateOnMatrix) {

  Matrix<float> m(2, 2);

  m(0, 0) = 1;
  m(0, 1) = 1;
  m(1, 0) = 1;
  m(1, 1) = 1;

  // Check that m.begin() returns a correct ptr
  // and that m.end() returns the correct end
  // by counting the number of element we iterate on
  size_t count = 0;
  for (float f : m) {
    ASSERT_EQ(1, f);
    count++;
  }
  ASSERT_EQ(4, count);
}

TEST(MatrixTest, ThrowOnInvalidMatrixMultiply) {
  Matrix<float> m(2, 9), n(1, 2);

  ASSERT_ANY_THROW(m * n);
}

TEST(MatrixTest, CanAddMatrix) {
  Matrix<float> m(2, 2), n(2, 2);

  m(0, 0) = 1;
  m(0, 1) = 1;
  m(1, 0) = 1;
  m(1, 1) = 1;

  n = m;

  auto c = m - n;
  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++)
      ASSERT_EQ(0, c(i, j));

  // Check that we can add in place

  m -= n;

  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++)
      ASSERT_EQ(0, c(i, j));
}

TEST(MatrixTest, CanSubMatrix) {
  Matrix<float> m(2, 2), n(2, 2);

  m(0, 0) = 1;
  m(0, 1) = 1;
  m(1, 0) = 1;
  m(1, 1) = 1;

  n = m;

  auto c = m - n;
  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++)
      ASSERT_EQ(0, c(i, j));

  // Check that we can add in place

  m -= n;

  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++)
      ASSERT_EQ(0, c(i, j));
}

TEST(MatrixTest, CanTransposeMatrix) {

  // Check we can transpose a square matrix
  Matrix<float> m(2, 2);

  m(0, 0) = 1;
  m(0, 1) = 0;
  m(1, 0) = 1;
  m(1, 1) = 0;

  auto t = m.transpose();
  ASSERT_EQ(2, t.getRows());
  ASSERT_EQ(2, t.getCols());

  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++) {
      ASSERT_EQ(m(j, i), t(i, j));
    }

  // Check we can transpose any matrix

  Matrix<float> n(3, 5);

  utils::random::randomize(n, 0.f, 100.f);

  t = n.transpose();
  ASSERT_EQ(5, t.getRows());
  ASSERT_EQ(3, t.getCols());

  for (size_t i = 0; i < 3; i++)
    for (size_t j = 0; j < 5; j++)
      ASSERT_EQ(n(i, j), t(j, i));
}

TEST(MatrixTest, CanMultiplyMatrix) {
  Matrix<float> m(2, 2), n(2, 2);

  m(0, 0) = 1;
  m(0, 1) = 1;
  m(1, 0) = 1;
  m(1, 1) = 1;

  n = m;

  auto c = m * n;
  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++)
      ASSERT_EQ(2, c(i, j));

  // Check that we can multiply matrix of different size (here, d is a vector)
  Matrix<float> d(2, 1);

  d(0, 0) = 1;
  d(1, 0) = 1;

  c = m * d;
  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 1; j++)
      ASSERT_EQ(2, c(i, j));
}

TEST(MatrixTest, ThrowOnInvalidSumReduce) {
  Matrix<float> m;

  ASSERT_ANY_THROW(m.sumReduce());
}

TEST(MatrixTest, CanSumReduce) {
  Matrix<float> m(2, 2);

  m(0, 0) = 1;
  m(0, 1) = 1;
  m(1, 0) = 1;
  m(1, 1) = 1;

  auto c = m.sumReduce();
  ASSERT_EQ(4, c);
}