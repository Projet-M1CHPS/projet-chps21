
#include "Utils.hpp"
#include "clUtils/clFMatrix.hpp"
#include <gtest/gtest.h>

using namespace math;
using namespace utils;

TEST(clFMatrixTest, CanCreate) {
  auto wrapper = clWrapper::makeDefault();
  clFMatrix m(10, 11, *wrapper);

  ASSERT_EQ(10, m.getRows());
  ASSERT_EQ(11, m.getCols());

  // Should be able to create an empty matrix
  const Matrix<float> n;

  ASSERT_EQ(0, n.getRows());
  ASSERT_EQ(0, n.getCols());
}


TEST(clFMatrixTest, CanCopy) {
  auto wrapper = clWrapper::makeDefault();
  FloatMatrix n(10, 11);

  n(0, 0) = 1;
  n(0, 1) = 1;
  n(1, 0) = 1;
  n(1, 1) = 1;

  // Should be able to copy from float matrix
  clFMatrix m(n, *wrapper);
  auto test = m.toFloatMatrix(*wrapper);

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(1, test(i, j));
  }

  // Should be able to copy two clFMatrices
  // Should be able to copy with constructor
  clFMatrix p(m, *wrapper);
  auto test2 = p.toFloatMatrix(*wrapper);

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(1, test(i, j));
  }

  // Should be able to copy an empty matrix
  clFMatrix o;
  m = clFMatrix(o, *wrapper);

  ASSERT_EQ(0, m.getRows());
  ASSERT_EQ(0, m.getCols());
}


TEST(clFMatrixTest, CanMoveCopy) {
  Matrix<float> m(2, 2);

  m(0, 0) = 1;
  m(0, 1) = 1;
  m(1, 0) = 1;
  m(1, 1) = 1;

  auto wrapper = clWrapper::makeDefault();
  clFMatrix n(m, *wrapper);

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
  clFMatrix m2(clFMatrix(3, 2, *wrapper), *wrapper);

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

  auto wrapper = clWrapper::makeDefault();
  clFMatrix a(m, *wrapper);
  clFMatrix b(n, *wrapper);
  auto buf = a.add(b, *wrapper);
  wrapper->getDefaultQueue().finish();

  auto c = buf.toFloatMatrix(*wrapper);

  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(m(i, j) + n(i, j), c(i, j));

  // Check that we can add in place
  buf = clFMatrix(m, *wrapper);
  buf.ipadd(b, *wrapper);
  wrapper->getDefaultQueue().finish();
  c = buf.toFloatMatrix(*wrapper);

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(m(i, j) + n(i, j), c(i, j));
  }
}

TEST(clFMatrixTest, ThrowOnInvalidAdd) {
  Matrix<float> m(2, 2), n(2, 1), o;
  auto wrapper = clWrapper::makeDefault();
  clFMatrix a(m, *wrapper), b(n, *wrapper), c(o, *wrapper);

  ASSERT_ANY_THROW(auto h = a.add(b, *wrapper));
  ASSERT_ANY_THROW(a.ipadd(b, *wrapper));
  ASSERT_ANY_THROW(auto d = a.add(c, *wrapper));
  ASSERT_ANY_THROW(a.ipadd(c, *wrapper));
}


TEST(clFMatrixTest, CanSubtract) {
  Matrix<float> m(2, 2), n(2, 2);

  m(0, 0) = 1;
  m(0, 1) = 1;
  m(1, 0) = 1;
  m(1, 1) = 2;

  n = m;
  n(1, 1) = 3;

  auto wrapper = clWrapper::makeDefault();
  clFMatrix a(m, *wrapper);
  clFMatrix b(n, *wrapper);
  auto buf = a.sub(b, *wrapper);
  wrapper->getDefaultQueue().finish();

  auto c = buf.toFloatMatrix(*wrapper);

  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(m(i, j) - n(i, j), c(i, j));

  // Check that we can add in place
  buf = clFMatrix(m, *wrapper);
  buf.ipsub(b, *wrapper);
  wrapper->getDefaultQueue().finish();
  c = buf.toFloatMatrix(*wrapper);

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(m(i, j) - n(i, j), c(i, j));
  }
}

TEST(clFMatrixTest, ThrowOnInvalidSubtraction) {
  Matrix<float> m(2, 2), n(2, 1), o;
  auto wrapper = clWrapper::makeDefault();
  clFMatrix a(m, *wrapper), b(n, *wrapper), c(o, *wrapper);

  ASSERT_ANY_THROW(auto h = a.sub(b, *wrapper));
  ASSERT_ANY_THROW(a.ipsub(b, *wrapper));
  ASSERT_ANY_THROW(auto d = a.sub(c, *wrapper));
  ASSERT_ANY_THROW(a.ipsub(c, *wrapper));
}

TEST(clFMatrixTest, CanScale) {
  Matrix<float> m(2, 2);
  const float scale = 2;

  m(0, 0) = 1;
  m(0, 1) = 2;
  m(1, 0) = 3;
  m(1, 1) = 4;
  auto wrapper = clWrapper::makeDefault();
  clFMatrix a(m, *wrapper);

  auto c = a.scale(scale, *wrapper);
  wrapper->getDefaultQueue().finish();
  auto n = c.toFloatMatrix(*wrapper);

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(m(i, j) * scale, n(i, j));
  }

  // Check that we can scale in place
  c = clFMatrix(a, *wrapper);
  c.ipscale(scale, *wrapper);
  wrapper->getDefaultQueue().finish();
  n = c.toFloatMatrix(*wrapper);


  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(m(i, j) * scale, n(i, j));
  }

  // Must be able to scale an empty matrix
  // without throwing error
  clFMatrix v(0, 0, *wrapper);
  auto x = v.scale(scale, *wrapper);
  v.ipscale(scale, *wrapper);
}

TEST(clFMatrixTest, CanTranspose) {
  Matrix<float> n(3, 5), o;

  randomize(n, 0.f, 100.f);
  auto wrapper = clWrapper::makeDefault();
  clFMatrix m(n, *wrapper);

  auto w = m.transpose(*wrapper);
  wrapper->getDefaultQueue().finish();

  auto t = w.toFloatMatrix(*wrapper);
  ASSERT_EQ(5, t.getRows());
  ASSERT_EQ(3, t.getCols());

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 5; j++) ASSERT_EQ(n(i, j), t(j, i));
  }

  // should be able to transpose an empty matrix without error
  w = clFMatrix(o, *wrapper).transpose(*wrapper);
  wrapper->getDefaultQueue().finish();

  auto u = w.toFloatMatrix(*wrapper);
  ASSERT_EQ(0, u.getRows());
  ASSERT_EQ(0, u.getCols());
}

TEST(clFMatrixTest, CanSumReduce) {
  Matrix<float> m(2, 2);

  m(0, 0) = 1;
  m(0, 1) = 1;
  m(1, 0) = 1;
  m(1, 1) = 1;

  auto wrapper = clWrapper::makeDefault();
  clFMatrix a(m, *wrapper);

  auto c = a.sumReduce(*wrapper);
  ASSERT_EQ(4, c);
}

TEST(clFMatrixTest, ThrowOnInvalidSumReduce) {
  Matrix<float> m;
  auto wrapper = clWrapper::makeDefault();
  clFMatrix a(m, *wrapper);


  ASSERT_ANY_THROW(a.sumReduce(*wrapper));
}


TEST(clFMatrixTest, CanSimpleGEMM) {
  Matrix<float> m(2, 2), n(2, 2);

  m(0, 0) = 1;
  m(0, 1) = 2;
  m(1, 0) = 1;
  m(1, 1) = 2;

  n = m;
  auto wrapper = clWrapper::makeDefault();
  clFMatrix a(m, *wrapper), b(m, *wrapper);

  auto c = clFMatrix::gemm(1.0f, false, a, false, b, *wrapper, true);
  Matrix<float> d(2, 2);

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) {
      float dij = 0;
      for (size_t k = 0; k < 2; k++) { dij += m(i, k) * n(k, j); }
      d(i, j) = dij;
    }
  }
  auto w = c.toFloatMatrix(*wrapper);

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(w(i, j), d(i, j));
  }
}

TEST(clFMatrixTest, ThrowOnInvalidGEMM) {
  Matrix<float> m(2, 2), n(1, 3);
  auto wrapper = clWrapper::makeDefault();
  clFMatrix a(m, *wrapper), b(n, *wrapper);

  ASSERT_ANY_THROW(auto c = clFMatrix::gemm(1.0f, false, a, false, b, *wrapper));
}

TEST(clFMatrixTest, CanHadamar) {
  Matrix<float> m(2, 3);

  m(0, 0) = 1;
  m(0, 1) = 2;
  m(0, 2) = 2;

  m(1, 0) = 4;
  m(1, 1) = 2;
  m(1, 2) = 3;

  auto wrapper = clWrapper::makeDefault();
  clFMatrix a(m, *wrapper), b(m, *wrapper);

  auto c = a.hadamard(b, *wrapper);
  wrapper->getDefaultQueue().finish();
  auto n = c.toFloatMatrix(*wrapper);

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) ASSERT_EQ(m(i, j) * m(i, j), n(i, j));
  }
}

TEST(clFMatrixTest, ThrowOnInvalidMatrixHadamardProd) {
  Matrix<float> m(2, 2), n(1, 3), o;
  auto wrapper = clWrapper::makeDefault();
  clFMatrix a(m, *wrapper), b(n, *wrapper), c(o, *wrapper);

  ASSERT_ANY_THROW(a.hadamard(b, *wrapper));
  ASSERT_ANY_THROW(c.hadamard(a, *wrapper));
  ASSERT_ANY_THROW(b.hadamard(c, *wrapper));
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

  auto wrapper = clWrapper::makeDefault();
  clFMatrix a(A, *wrapper), b(B, *wrapper), c(C, *wrapper);

  auto res = clFMatrix::gemm(1.0f, false, a, false, b, 1.0f, c, *wrapper);
  auto l = res.toFloatMatrix(*wrapper);

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 1; j++) ASSERT_EQ(D(i, j), l(i, j));
  }
}

TEST(clFMatrixTest, ThrowOnInvalidMatMatProdMatAdd) {
  auto wrapper = clWrapper::makeDefault();
  clFMatrix a(2, 2, *wrapper), b(1, 3, *wrapper), c(2, 3, *wrapper), o;

  ASSERT_ANY_THROW(clFMatrix::gemm(1.0f, false, a, false, b, 1.0f, c, *wrapper));
  ASSERT_ANY_THROW(clFMatrix::gemm(1.0f, false, a, false, b, 1.0f, o, *wrapper));
  ASSERT_ANY_THROW(clFMatrix::gemm(1.0f, false, a, false, o, 1.0f, c, *wrapper));
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

  auto wrapper = clWrapper::makeDefault();
  clFMatrix a(A, *wrapper), b(B, *wrapper), c(C, *wrapper);


  auto D = clFMatrix::gemm(2.f, false, a, false, b, *wrapper);
  auto E = clFMatrix::gemm(3.f, true, a, true, b, *wrapper);
  auto F = clFMatrix::gemm(4.f, true, a, false, c, *wrapper);
  auto G = clFMatrix::gemm(1.0f, false, a, true, c, *wrapper);

  auto d_fmat = D.toFloatMatrix(*wrapper);
  auto e_fmat = E.toFloatMatrix(*wrapper);
  auto f_fmat = F.toFloatMatrix(*wrapper);
  auto g_fmat = G.toFloatMatrix(*wrapper);

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
  auto wrapper = clWrapper::makeDefault();
  clFMatrix A(3, 2, *wrapper), B(2, 3, *wrapper), C(3, 2, *wrapper), o;

  ASSERT_ANY_THROW(clFMatrix::gemm(1.0f, false, A, false, C, *wrapper));
  ASSERT_ANY_THROW(clFMatrix::gemm(1.f, true, A, true, C, *wrapper));
  ASSERT_ANY_THROW(clFMatrix::gemm(1.0f, true, A, false, B, *wrapper));
  ASSERT_ANY_THROW(clFMatrix::gemm(2.f, false, A, true, B, *wrapper));
  ASSERT_ANY_THROW(clFMatrix::gemm(3.f, false, A, true, o, *wrapper));
}