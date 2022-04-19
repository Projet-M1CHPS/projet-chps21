#include "Utils.hpp"
#include "math/clFMatrix.hpp"
#include "math/clFTensor.hpp"
#include <gtest/gtest.h>

using namespace math;
using namespace utils;

TEST(clFTensor, canCreateEmpty) {
  clFTensor t;
  EXPECT_EQ(t.getDepth(), 0);
  EXPECT_EQ(t.getRows(), 0);
  EXPECT_EQ(t.getCols(), 0);
}

TEST(clFTensor, canCreate) {
  clFTensor t2(1, 2, 3);

  EXPECT_EQ(t2.getRows(), 1);
  EXPECT_EQ(t2.getCols(), 2);
  EXPECT_EQ(t2.getDepth(), 3);
}

TEST(clFTensor, canFlatten) {
  clFTensor t;
  auto flat = t.flatten();
  EXPECT_EQ(flat.getRows(), 0);
  EXPECT_EQ(flat.getCols(), 0);
  EXPECT_EQ(flat.getDepth(), 0);

  clFTensor t2(1, 2, 3);
  flat = t2.flatten();

  EXPECT_EQ(flat.getRows(), 2);
  EXPECT_EQ(flat.getCols(), 1);
  EXPECT_EQ(flat.getDepth(), 3);
}

TEST(clFTensor, canDeepCopy) {
  clFTensor t(1, 2, 3);
  clFTensor t2(t, true);

  EXPECT_EQ(t2.getRows(), 1);
  EXPECT_EQ(t2.getCols(), 2);
  EXPECT_EQ(t2.getDepth(), 3);

  // The buffer should not have been copied
  EXPECT_NE(t2.getBuffer()(), t.getBuffer()());
}

TEST(clFTensor, canShallowCopy) {
  clFTensor t(1, 2, 3);
  clFTensor t2 = t.shallowCopy();

  EXPECT_EQ(t2.getRows(), 1);
  EXPECT_EQ(t2.getCols(), 2);
  EXPECT_EQ(t2.getDepth(), 3);

  // The buffer should have been copied
  EXPECT_EQ(t2.getBuffer()(), t.getBuffer()());
}

TEST(clFTensor, canCreateSubmatrix) {
  FloatMatrix m(1, 2);
  m(0, 0) = 1;
  m(0, 1) = 1;

  clFTensor tensor(1, 2, 3);
  clFMatrix mat1 = tensor[0];

  EXPECT_EQ(mat1.getRows(), 1);
  EXPECT_EQ(mat1.getCols(), 2);
  mat1 = m;
  mat1.ipadd(1.0f, mat1);

  clFMatrix mat2 = tensor[1];
  EXPECT_EQ(mat2.getRows(), 1);
  EXPECT_EQ(mat2.getCols(), 2);
  mat2 = m;
  mat2.ipadd(2.0f, mat2);

  clFMatrix mat3 = tensor[2];
  EXPECT_EQ(mat3.getRows(), 1);
  EXPECT_EQ(mat3.getCols(), 2);
  mat3 = m;
  mat3.ipadd(5.0f, mat3);

  utils::cl_wrapper.getDefaultQueue().finish();

  clFMatrix entire_tensor(tensor.getBuffer(), 6, 1, 0);
  auto fmatrix = entire_tensor.toFloatMatrix();

  EXPECT_EQ(fmatrix.getRows(), 6);
  EXPECT_EQ(fmatrix.getCols(), 1);

  FloatMatrix expected(6, 1);
  expected(0, 0) = 2;
  expected(1, 0) = 2;
  expected(2, 0) = 3;
  expected(3, 0) = 3;
  expected(4, 0) = 6;
  expected(5, 0) = 6;
  for (size_t i = 0; i < 6; i++) { EXPECT_EQ(fmatrix(i, 0), expected(i, 0)); }
}

TEST(clFTensor, canSplitShallowCopy) {
  FloatMatrix m(1, 2);
  m(0, 0) = 1;
  m(0, 1) = 1;


  clFTensor tensor(1, 2, 3);
  auto subtensors = tensor.slice(3);

  for (size_t i = 0; i < 3; i++) {
    EXPECT_EQ(subtensors[i].getRows(), 1);
    EXPECT_EQ(subtensors[i].getCols(), 2);
    EXPECT_EQ(subtensors[i].getDepth(), 1);
  }

  clFMatrix mat1 = subtensors[0][0];

  EXPECT_EQ(mat1.getRows(), 1);
  EXPECT_EQ(mat1.getCols(), 2);
  mat1 = m;
  mat1.ipadd(1.0f, mat1);

  clFMatrix mat2 = subtensors[1][0];
  EXPECT_EQ(mat2.getRows(), 1);
  EXPECT_EQ(mat2.getCols(), 2);
  mat2 = m;
  mat2.ipadd(2.0f, mat2);

  clFMatrix mat3 = subtensors[2][0];
  EXPECT_EQ(mat3.getRows(), 1);
  EXPECT_EQ(mat3.getCols(), 2);
  mat3 = m;
  mat3.ipadd(5.0f, mat3);

  utils::cl_wrapper.getDefaultQueue().finish();

  clFMatrix entire_tensor(tensor.getBuffer(), 6, 1, 0);
  auto fmatrix = entire_tensor.toFloatMatrix();

  EXPECT_EQ(fmatrix.getRows(), 6);
  EXPECT_EQ(fmatrix.getCols(), 1);

  FloatMatrix expected(6, 1);
  expected(0, 0) = 2;
  expected(1, 0) = 2;
  expected(2, 0) = 3;
  expected(3, 0) = 3;
  expected(4, 0) = 6;
  expected(5, 0) = 6;
  for (size_t i = 0; i < 6; i++) { EXPECT_EQ(fmatrix(i, 0), expected(i, 0)); }
}


TEST(clFTensor, canSubtract) {
  FloatMatrix m(5, 5);
  clFTensor tensor_a(5, 5, 10);
  clFTensor tensor_b(5, 5, 10);

  std::vector<FloatMatrix> as;
  std::vector<FloatMatrix> bs;

  for (auto &mat : tensor_a.getMatrices()) {
    randomize(m, 0.0f, 100.0f);
    as.push_back(m);
    mat = m;
  }

  for (auto &mat : tensor_b.getMatrices()) {
    randomize(m, 0.0f, 100.0f);
    bs.push_back(m);
    mat = m;
  }

  cl::CommandQueue queue = utils::cl_wrapper.getDefaultQueue();
  clFTensor sub = tensor_a.sub(1.0f, tensor_b, queue, true);

  for (size_t i = 0; auto &mat : sub.getMatrices()) {
    FloatMatrix tmp = mat.toFloatMatrix();
    FloatMatrix exact = as[i] - bs[i];

    for (size_t j = 0; j < exact.getSize(); j++) {
      EXPECT_EQ(tmp.getData()[j], exact.getData()[j]);
    }
    i++;
  }
}

TEST(clFTensor, canHadamard) {
  FloatMatrix m(5, 5);
  clFTensor tensor(5, 5, 10);

  std::vector<FloatMatrix> a;
  for (auto &mat : tensor.getMatrices()) {
    randomize(m, 0.0f, 100.f);
    mat = m;
    a.push_back(m);
  }

  cl::CommandQueue queue = utils::cl_wrapper.getDefaultQueue();
  tensor.iphadamard(tensor, queue, true);

  for (size_t i = 0; auto &mat : tensor.getMatrices()) {
    FloatMatrix tmp = mat.toFloatMatrix();
    FloatMatrix exact = a[i];
    exact.hadamardProd(exact);

    for (size_t j = 0; j < exact.getSize(); j++) {
      EXPECT_EQ(tmp.getData()[j], exact.getData()[j]);
    }
    i++;
  }
}

TEST(clFTensor, canSumCollapse) {
  FloatMatrix m(5, 5);
  clFTensor tensor(5, 5, 100);

  std::vector<FloatMatrix> a;
  for (auto &mat : tensor.getMatrices()) {
    randomize(m, 0.0f, 100.f);
    mat = m;
    a.push_back(m);
  }

  cl::CommandQueue queue = utils::cl_wrapper.getDefaultQueue();
  FloatMatrix tmp = tensor.sumCollapse(queue, true).toFloatMatrix();
  FloatMatrix exact(5, 5);
  exact.fill(0);
  for (auto &buf : a) { exact += buf; }
  for (size_t j = 0; j < exact.getSize(); j++) {
    EXPECT_NEAR(tmp.getData()[j], exact.getData()[j], 0.1);
  }
}

// where _t denotes a tensor :
// R = A * B_t + C
TEST(clFTensor, canBatchGemmMatrixTensorMatrix) {
  FloatMatrix a(5, 5), c(5, 4);
  randomize(a, 0.f, 1.f);
  randomize(c, 0.f, 1.f);
  clFTensor tensor(5, 4, 10);
  clFMatrix cla(a);
  clFMatrix clc(c);


  std::vector<FloatMatrix> b;
  for (auto &mat : tensor.getMatrices()) {
    FloatMatrix buf(5, 4);
    randomize(buf, 0.0f, 100.f);
    mat = buf;
    b.push_back(buf);
  }

  cl::CommandQueue queue = utils::cl_wrapper.getDefaultQueue();
  clFTensor res =
          math::clFTensor::batchedGemm(1.0f, false, cla, false, tensor, 1.0f, clc, queue, true);

  for (size_t i = 0; i < res.getDepth(); i++) {
    FloatMatrix exact = math::FloatMatrix::matMatProdMatAdd(a, b[i], c);
    FloatMatrix tmp = res[i].toFloatMatrix();
    for (size_t j = 0; j < exact.getSize(); j++) {
      EXPECT_FLOAT_EQ(tmp.getData()[j], exact.getData()[j]);
    }
  }
}

// where _t denotes a tensor :
// R = A_t * B
TEST(clFTensor, canBatchGemmMatrixTensor) {
  FloatMatrix a(5, 5);
  randomize(a, 0.f, 1.f);
  clFTensor tensor(5, 4, 10);
  clFMatrix cla(a);


  std::vector<FloatMatrix> b;
  for (auto &mat : tensor.getMatrices()) {
    FloatMatrix buf(5, 4);
    randomize(buf, 0.0f, 100.f);
    mat = buf;
    b.push_back(buf);
  }

  cl::CommandQueue queue = utils::cl_wrapper.getDefaultQueue();
  clFTensor res = math::clFTensor::batchedGemm(1.0f, false, cla, false, tensor, queue, true);

  for (size_t i = 0; i < res.getDepth(); i++) {
    FloatMatrix exact = math::FloatMatrix::mul(false, a, false, b[i]);
    FloatMatrix tmp = res[i].toFloatMatrix();
    for (size_t j = 0; j < exact.getSize(); j++) {
      EXPECT_FLOAT_EQ(tmp.getData()[j], exact.getData()[j]);
    }
  }
}
// where _t denotes a tensor :
// R = A_t * B_t
TEST(clFTensor, canBatchGemmTensorTensor) {
  clFTensor tensor_a(10, 50, 10);
  clFTensor tensor_b(50, 8, 10);


  std::vector<FloatMatrix> a;
  for (auto &mat : tensor_a.getMatrices()) {
    FloatMatrix buf(10, 50);
    randomize(buf, 0.0f, 1.f);
    mat = buf;
    a.push_back(buf);
  }

  std::vector<FloatMatrix> b;
  for (auto &mat : tensor_b.getMatrices()) {
    FloatMatrix buf(50, 8);
    randomize(buf, 0.0f, 1.f);
    mat = buf;
    b.push_back(buf);
  }


  cl::CommandQueue queue = utils::cl_wrapper.getDefaultQueue();
  clFTensor res = math::clFTensor::batchedGemm(1.0f, false, tensor_a, false, tensor_b, queue, true);

  for (size_t i = 0; i < res.getDepth(); i++) {
    FloatMatrix exact = math::FloatMatrix::mul(false, a[i], false, b[i]);
    FloatMatrix tmp = res[i].toFloatMatrix();
    for (size_t j = 0; j < exact.getSize(); j++) {
      EXPECT_NEAR(tmp.getData()[j], exact.getData()[j], 0.1);
    }
  }
}
