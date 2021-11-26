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

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(1, n(i, j));
  }
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

  n = std::move(m1);
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

TEST(MatrixTest, CanAddMatrix) {
  Matrix<float> m(2, 2), n(2, 2);

  m(0, 0) = 1;
  m(0, 1) = 1;
  m(1, 0) = 1;
  m(1, 1) = 2;

  n = m;
  n(1, 1) = 3;

  auto c = m + n;
  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(m(i, j) + n(i, j), c(i, j));

  // Check that we can sub in place
  c = m;
  c += n;

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(m(i, j) + n(i, j), c(i, j));
  }
}

TEST(MatrixTest, ThrowOnInvalidMatrixAdd) {
  Matrix<float> m(2, 2), n(2, 1);

  ASSERT_ANY_THROW(auto c = m + n);
  ASSERT_ANY_THROW(m += n);
}

TEST(MatrixTest, CanSubMatrix) {
  Matrix<float> m(2, 2), n(2, 2);

  m(0, 0) = 1;
  m(0, 1) = 2;
  m(1, 0) = 1;
  m(1, 1) = 2;

  n = m;
  n(1, 1) = 3;

  auto c = m - n;
  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(m(i, j) - n(i, j), c(i, j));

  // Check that we can sub in place
  c = m;
  c -= n;

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(m(i, j) - n(i, j), c(i, j));
  }
}

TEST(MatrixTest, ThrowOnInvalidMatrixSub) {
  Matrix<float> m(2, 2), n(2, 1);

  ASSERT_ANY_THROW(auto c = m - n);
  ASSERT_ANY_THROW(m -= n);
}

TEST(MatrixTest, CanTransposeMatrix) {
  Matrix<float> n(3, 5);

  utils::random::randomize(n, 0.f, 100.f);

  auto t = n.transpose();
  ASSERT_EQ(5, t.getRows());
  ASSERT_EQ(3, t.getCols());

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 5; j++) ASSERT_EQ(n(i, j), t(j, i));
  }
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
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(2, c(i, j));

  // Check that we can multiply matrix of different size (here, d is a vector)
  Matrix<float> d(2, 1);

  d(0, 0) = 1;
  d(1, 0) = 1;

  c = m * d;
  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 1; j++) ASSERT_EQ(2, c(i, j));
  }
}

TEST(MatrixTest, ThrowOnInvalidMatrixMultiply) {
  Matrix<float> m(2, 9), n(1, 2);

  ASSERT_ANY_THROW(m * n);
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

TEST(MatrixTest, ThrowOnInvalidSumReduce) {
  Matrix<float> m;

  ASSERT_ANY_THROW(m.sumReduce());
}

TEST(MatrixTest, CanMulMatrixWithMatrix) {
  Matrix<float> m(2, 2), n(2, 2);

  m(0, 0) = 1;
  m(0, 1) = 2;
  m(1, 0) = 1;
  m(1, 1) = 2;

  n = m;
  n(1, 1) = 3;

  auto c = m * n;
  Matrix<float> d(2, 2);

  for (size_t i = 0; i < 2; i++) {
    for (size_t k = 0; k < 2; k++) {
      for (size_t j = 0; j < 2; j++) { d(i, j) += m(i, k) * n(k, j); }
    }
  }

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(c(i, j), d(i, j));
  }
}

TEST(MatrixTest, ThrowOnInvalidMatrixMul) {
  Matrix<float> m(2, 2), n(1, 3);

  ASSERT_ANY_THROW(auto c = m * n);
}

TEST(MatrixTest, CanMulMatrixWithScale) {
  Matrix<float> m(2, 2);
  const float scale{2};

  m(0, 0) = 1;
  m(0, 1) = 2;
  m(1, 0) = 3;
  m(1, 1) = 4;

  auto c = m * scale;
  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(m(i, j) * scale, c(i, j));
  }

  // Check that we can mul in place
  c = m;
  c *= scale;

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(m(i, j) * scale, c(i, j));
  }
}

TEST(MatrixTest, CanHadamardMulMatrix) {
  Matrix<float> m(2, 3), n(2, 3);

  m(0, 0) = 1;
  m(0, 1) = 2;
  m(0, 2) = 2;

  m(1, 0) = 4;
  m(1, 1) = 2;
  m(1, 2) = 3;

  n = m;
  n(1, 1) = 3;

  auto c = m;
  c.hadamardProd(n);

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(m(i, j) * n(i, j), c(i, j));
  }
}

TEST(MatrixTest, ThrowOnInvalidMatrixHadamardMul) {
  Matrix<float> m(2, 2), n(1, 3);

  ASSERT_ANY_THROW(m.hadamardProd(n));
}


TEST(MatrixTest, CanMatMatProdMatAdd) {
  Matrix<float> A(2, 3), B(3, 1), C(2, 1);

  A(0, 0) = 1;
  A(0, 1) = 4;
  A(0, 2) = 2;
  A(1, 0) = 1;
  A(1, 1) = 2;
  A(1, 2) = 5;

  B(1, 0) = 4;
  B(2, 0) = 2;
  B(3, 0) = 4;

  C(1, 0) = 5;
  C(2, 0) = 3;

  Matrix<float> D(2, 1);

  for (size_t i = 0; i < A.getRows(); i++) {
    for (size_t k = 0; k < A.getCols(); k++) {
      for (size_t j = 0; j < B.getCols(); j++) { D(i, j) += A(i, k) * B(k, j); }
    }
  }

  for (size_t i = 0; i < C.getRows(); i++) {
    for (size_t j = 0; j < C.getCols(); j++) { D(i, j) += C(i, j); }
  }

  auto res = Matrix<float>::matMatProdMatAdd(A, B, C);

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(D(i, j), res(i, j));
  }
}

TEST(MatrixTest, ThrowOnInvalidMatrixMatMatProdMatAdd) {
  Matrix<float> A(2, 2), B(1, 3), C(2, 3);

  ASSERT_ANY_THROW(Matrix<float>::matMatProdMatAdd(A, B, C));
}


TEST(MatrixTest, CanMatMatProd) {
  Matrix<float> A(3, 2), B(2, 3), C(3, 2);

  A(0, 0) = 1;
  A(0, 1) = 2;
  A(1, 0) = 3;
  A(1, 1) = 4;
  A(2, 0) = 12;
  A(2, 1) = 8;

  B(0, 0) = 4;
  B(0, 1) = 1;
  B(0, 2) = 12;
  B(1, 0) = 2;
  B(1, 1) = 6;
  B(1, 2) = 9;

  C(0, 0) = 1;
  C(0, 1) = 2;
  C(1, 0) = 3;
  C(1, 1) = 4;
  C(2, 0) = 12;
  C(2, 1) = 8;

  Matrix<float> D = Matrix<float>::mul(false, A, false, B, 2.f);
  Matrix<float> E = Matrix<float>::mul(true, A, true, B, 3.f);
  Matrix<float> F = Matrix<float>::mul(true, A, false, C, 4.f);
  Matrix<float> G = Matrix<float>::mul(false, A, true, C);

  Matrix<float> d = A * 2.f * B;
  Matrix<float> e = A.transpose() * 3.f * B.transpose();
  Matrix<float> f = A.transpose() * 4.f * C;
  Matrix<float> g = A * C.transpose();

  for (size_t i = 0; i < D.getCols(); i++) {
    for (size_t j = 0; j < D.getRows(); j++) { ASSERT_EQ(d(i, j), D(i, j)); }
  }

  for (size_t i = 0; i < E.getCols(); i++) {
    for (size_t j = 0; j < E.getRows(); j++) { ASSERT_EQ(e(i, j), E(i, j)); }
  }

  for (size_t i = 0; i < F.getCols(); i++) {
    for (size_t j = 0; j < F.getRows(); j++) { ASSERT_EQ(f(i, j), F(i, j)); }
  }

  for (size_t i = 0; i < G.getCols(); i++) {
    for (size_t j = 0; j < G.getRows(); j++) { ASSERT_EQ(g(i, j), G(i, j)); }
  }
}

TEST(MatrixTest, ThrowOnInvalidMatrixMatMatProd) {
  Matrix<float> A(3, 2), B(2, 3), C(3, 2);

  ASSERT_ANY_THROW(Matrix<float>::mul(false, A, false, C));
  ASSERT_ANY_THROW(Matrix<float>::mul(true, A, true, C, 3.f));
  ASSERT_ANY_THROW(Matrix<float>::mul(true, A, false, B));
  ASSERT_ANY_THROW(Matrix<float>::mul(false, A, true, B, 1.f));
}


TEST(MatrixTest, CanMatTransMatProd) {
  Matrix<float> A(3, 2), B(3, 2);

  A(0, 0) = 1;
  A(0, 1) = 2;
  A(1, 0) = 3;
  A(1, 1) = 4;
  A(2, 0) = 12;
  A(2, 1) = 8;

  B(0, 0) = 1;
  B(0, 1) = 2;
  B(1, 0) = 3;
  B(1, 1) = 4;
  B(2, 0) = 12;
  B(2, 1) = 8;

  Matrix<float> C = Matrix<float>::matTransMatProd(A, B);

  Matrix<float> c = A.transpose() * B;

  for (size_t i = 0; i < C.getCols(); i++) {
    for (size_t j = 0; j < C.getRows(); j++) { ASSERT_EQ(c(i, j), C(i, j)); }
  }
}

TEST(MatrixTest, ThrowOnInvalidMatrixMatTransMatProd) {
  Matrix<float> A(3, 2), B(2, 3);

  ASSERT_ANY_THROW(Matrix<float>::matTransMatProd(A, B));
}

TEST(MatrixTest, CanMatMatTransProd) {
  Matrix<float> A(3, 2), B(3, 2);

  A(0, 0) = 1;
  A(0, 1) = 2;
  A(1, 0) = 3;
  A(1, 1) = 4;
  A(2, 0) = 12;
  A(2, 1) = 8;

  B(0, 0) = 1;
  B(0, 1) = 2;
  B(1, 0) = 3;
  B(1, 1) = 4;
  B(2, 0) = 12;
  B(2, 1) = 8;

  Matrix<float> C = Matrix<float>::matMatTransProd(A, B);

  Matrix<float> c = A * B.transpose();

  for (size_t i = 0; i < C.getCols(); i++) {
    for (size_t j = 0; j < C.getRows(); j++) { ASSERT_EQ(c(i, j), C(i, j)); }
  }
}

TEST(MatrixTest, ThrowOnInvalidMatrixMatMatTransProd) {
  Matrix<float> A(3, 2), B(2, 3);

  ASSERT_ANY_THROW(Matrix<float>::matMatTransProd(A, B));
}