
#include "Utils.hpp"
#include "math/clFMatrix.hpp"
#include <gtest/gtest.h>

using namespace math;
using namespace utils;

TEST(clFMatrixTest, CanCreate) {
  clFMatrix m(10, 11);

  ASSERT_EQ(10, m.getRows());
  ASSERT_EQ(11, m.getCols());

  // Should be able to create an empty matrix
  const clFMatrix n;

  ASSERT_EQ(0, n.getRows());
  ASSERT_EQ(0, n.getCols());
}


TEST(clFMatrixTest, CanCopy) {
  FloatMatrix n(10, 11);

  n(0, 0) = 1;
  n(0, 1) = 1;
  n(1, 0) = 1;
  n(1, 1) = 1;

  // Should be able to copy from float matrix
  clFMatrix m(n);

  auto test = m.toFloatMatrix();

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(1, test(i, j));
  }

  // Should be able to copy two clFMatrices
  clFMatrix p;
  p.copy(m, true);
  auto test2 = p.toFloatMatrix();

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(1, test(i, j));
  }

  // Should be able to copy an empty matrix
  clFMatrix o;
  m.copy(o, true);

  ASSERT_EQ(0, m.getRows());
  ASSERT_EQ(0, m.getCols());
}


TEST(clFMatrixTest, CanMoveCopy) {
  Matrix<float> m(2, 2);

  m(0, 0) = 1;
  m(0, 1) = 1;
  m(1, 0) = 1;
  m(1, 1) = 1;

  clFMatrix n(m);

  auto raw_ptr = n.getBuffer();

  clFMatrix c = std::move(n);
  // The move_copy should not reallocate an arrayd
  ASSERT_EQ(raw_ptr, c.getBuffer());
  ASSERT_EQ(2, c.getCols());
  ASSERT_EQ(2, c.getRows());

  // should be able to copy an empty matrix
  clFMatrix m1;

  c = std::move(m1);
  // The src matrix should be reset to an empty matrix
  ASSERT_EQ(0, c.getRows());
  ASSERT_EQ(0, c.getCols());

  // should be able to copy a right value matrix
  clFMatrix m2(clFMatrix(3, 2));

  ASSERT_EQ(3, m2.getRows());
  ASSERT_EQ(2, m2.getCols());
}

TEST(clFMatrixTest, CanAdd) {
  Matrix<float> m(2, 2), n(2, 2);

  m(0, 0) = 1;
  m(0, 1) = 1;
  m(1, 0) = 1;
  m(1, 1) = 2;

  n = m;
  n(1, 1) = 3;

  clFMatrix a(m);
  clFMatrix b(n);
  auto buf = a.add(1.0f, b);
  auto c = buf.toFloatMatrix();

  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(m(i, j) + n(i, j), c(i, j));

  // Check that we can add in place
  buf = m;
  buf.ipadd(1.0f, b);
  c = buf.toFloatMatrix();

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(m(i, j) + n(i, j), c(i, j));
  }
}

TEST(clFMatrixTest, ThrowOnInvalidAdd) {
  clFMatrix a(2, 2), b(2, 1), c;

  ASSERT_ANY_THROW(auto h = a.add(1.0f, b));
  ASSERT_ANY_THROW(a.ipadd(1.0f, b));
  ASSERT_ANY_THROW(auto d = a.add(1.0f, c));
  ASSERT_ANY_THROW(a.ipadd(1.0f, c));
}


TEST(clFMatrixTest, CanSubtract) {
  Matrix<float> m(2, 2), n(2, 2);

  m(0, 0) = 1;
  m(0, 1) = 1;
  m(1, 0) = 1;
  m(1, 1) = 2;

  n = m;
  n(1, 1) = 3;

  clFMatrix a(m);
  clFMatrix b(n);
  auto buf = a.sub(1.0f, b);
  auto c = buf.toFloatMatrix();

  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(m(i, j) - n(i, j), c(i, j));

  // Check that we can add in place
  buf = clFMatrix(m);
  buf.ipsub(1.0f, b);
  c = buf.toFloatMatrix();

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(m(i, j) - n(i, j), c(i, j));
  }
}

TEST(clFMatrixTest, ThrowOnInvalidSubtraction) {
  clFMatrix a(2, 2), b(2, 1), c;

  ASSERT_ANY_THROW(auto h = a.sub(1.0f, b));
  ASSERT_ANY_THROW(a.ipsub(1.0f, b));
  ASSERT_ANY_THROW(auto d = a.sub(1.0f, c));
  ASSERT_ANY_THROW(a.ipsub(1.0f, c));
}

TEST(clFMatrixTest, CanScale) {
  Matrix<float> m(2, 2);
  const float scale = 2;

  m(0, 0) = 1;
  m(0, 1) = 2;
  m(1, 0) = 3;
  m(1, 1) = 4;
  clFMatrix a(m);

  auto c = a.scale(scale);
  auto n = c.toFloatMatrix();

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(m(i, j) * scale, n(i, j));
  }

  // Check that we can scale in place
  c.copy(a, false);
  c.ipscale(scale);
  n = c.toFloatMatrix();


  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(m(i, j) * scale, n(i, j));
  }

  // Must be able to scale an empty matrix
  // without throwing error
  clFMatrix v(0, 0);
  auto x = v.scale(scale);
  v.ipscale(scale);
}

TEST(clFMatrixTest, CanTranspose) {
  Matrix<float> n(3, 5);

  randomize(n, 0.f, 100.f);
  clFMatrix m(n);

  auto w = m.transpose();
  auto t = w.toFloatMatrix();

  ASSERT_EQ(5, t.getRows());
  ASSERT_EQ(3, t.getCols());

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 5; j++) ASSERT_EQ(n(i, j), t(j, i));
  }

  // should be able to transpose an empty matrix without error
  w = clFMatrix().transpose();

  auto u = w.toFloatMatrix();
  ASSERT_EQ(0, u.getRows());
  ASSERT_EQ(0, u.getCols());
}

TEST(clFMatrixTest, CanSumReduce) {
  Matrix<float> m(2, 2);

  m(0, 0) = 1;
  m(0, 1) = 1;
  m(1, 0) = 1;
  m(1, 1) = 1;

  clFMatrix a(m);

  auto c = a.sumReduce();
  ASSERT_EQ(4, c);
}

TEST(clFMatrixTest, ThrowOnInvalidSumReduce) {
  clFMatrix a;

  ASSERT_ANY_THROW(a.sumReduce());
}


TEST(clFMatrixTest, CanSimpleGEMM) {
  Matrix<float> m(2, 2), n(2, 2);

  m(0, 0) = 1;
  m(0, 1) = 2;
  m(1, 0) = 1;
  m(1, 1) = 2;

  n = m;
  clFMatrix a(m), b(m);

  auto c = clFMatrix::gemm(1.0f, false, a, false, b, true);
  Matrix<float> d(2, 2);

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) {
      float dij = 0;
      for (size_t k = 0; k < 2; k++) { dij += m(i, k) * n(k, j); }
      d(i, j) = dij;
    }
  }
  auto w = c.toFloatMatrix();

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(w(i, j), d(i, j));
  }
}

TEST(clFMatrixTest, ThrowOnInvalidGEMM) {
  Matrix<float> m(2, 2), n(1, 3);
  clFMatrix a(m), b(n);

  ASSERT_ANY_THROW(auto c = clFMatrix::gemm(1.0f, false, a, false, b));
}

TEST(clFMatrixTest, CanHadamar) {
  Matrix<float> m(2, 3);

  m(0, 0) = 1;
  m(0, 1) = 2;
  m(0, 2) = 2;

  m(1, 0) = 4;
  m(1, 1) = 2;
  m(1, 2) = 3;

  clFMatrix a(m), b(m);

  auto c = a.hadamard(b);
  auto n = c.toFloatMatrix();

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(m(i, j) * m(i, j), n(i, j));
  }
}

TEST(clFMatrixTest, ThrowOnInvalidMatrixHadamardProd) {
  Matrix<float> m(2, 2), n(1, 3), o;
  clFMatrix a(m), b(n), c(o);

  ASSERT_ANY_THROW(a.hadamard(b));
  ASSERT_ANY_THROW(c.hadamard(a));
  ASSERT_ANY_THROW(b.hadamard(c));
}

TEST(clFMatrixTest, CanComplexGEMM) {
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

  D = A * B;
  D += C;

  clFMatrix a(A), b(B), c(C);

  auto res = clFMatrix::gemm(1.0f, false, a, false, b, 1.0f, c);
  auto l = res.toFloatMatrix();

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 1; j++) ASSERT_EQ(D(i, j), l(i, j));
  }
}

TEST(clFMatrixTest, ThrowOnInvalidMatMatProdMatAdd) {
  clFMatrix a(2, 2), b(1, 3), c(2, 3), o;

  ASSERT_ANY_THROW(clFMatrix::gemm(1.0f, false, a, false, b, 1.0f, c));
  ASSERT_ANY_THROW(clFMatrix::gemm(1.0f, false, a, false, b, 1.0f, o));
  ASSERT_ANY_THROW(clFMatrix::gemm(1.0f, false, a, false, o, 1.0f, c));
}


TEST(clFMatrixTest, CanGEMMWithTranspose) {
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

  clFMatrix a(A), b(B), c(C);


  auto D = clFMatrix::gemm(2.f, false, a, false, b);
  auto E = clFMatrix::gemm(3.f, true, a, true, b);
  auto F = clFMatrix::gemm(4.f, true, a, false, c);
  auto G = clFMatrix::gemm(1.0f, false, a, true, c);

  auto d_fmat = D.toFloatMatrix();
  auto e_fmat = E.toFloatMatrix();
  auto f_fmat = F.toFloatMatrix();
  auto g_fmat = G.toFloatMatrix();

  Matrix<float> d = A * 2.f * B;
  Matrix<float> e = A.transpose() * 3.f * B.transpose();
  Matrix<float> f = A.transpose() * 4.f * C;
  Matrix<float> g = A * C.transpose();

  for (size_t i = 0; i < D.getCols(); i++) {
    for (size_t j = 0; j < D.getRows(); j++) { ASSERT_EQ(d(i, j), d_fmat(i, j)); }
  }

  for (size_t i = 0; i < E.getCols(); i++) {
    for (size_t j = 0; j < E.getRows(); j++) { ASSERT_EQ(e(i, j), e_fmat(i, j)); }
  }

  for (size_t i = 0; i < F.getCols(); i++) {
    for (size_t j = 0; j < F.getRows(); j++) { ASSERT_EQ(f(i, j), f_fmat(i, j)); }
  }

  for (size_t i = 0; i < G.getCols(); i++) {
    for (size_t j = 0; j < G.getRows(); j++) { ASSERT_EQ(g(i, j), g_fmat(i, j)); }
  }
}

TEST(clFMatrixTest, ThrowOnInvalidGEMMWithTranspose) {
  clFMatrix A(3, 2), B(2, 3), C(3, 2), o;

  ASSERT_ANY_THROW(clFMatrix::gemm(1.0f, false, A, false, C));
  ASSERT_ANY_THROW(clFMatrix::gemm(1.f, true, A, true, C));
  ASSERT_ANY_THROW(clFMatrix::gemm(1.0f, true, A, false, B));
  ASSERT_ANY_THROW(clFMatrix::gemm(2.f, false, A, true, B));
  ASSERT_ANY_THROW(clFMatrix::gemm(3.f, false, A, true, o));
}

int main(int argc, char **argv) {
  utils::clWrapper::initOpenCL(*utils::clWrapper::makeDefault("../kernels"));
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}