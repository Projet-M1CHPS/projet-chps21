#include "Utils.hpp"
#include "clUtils/clFMatrix.hpp"
#include "clUtils/clFTensor.hpp"
#include <gtest/gtest.h>

using namespace math;
using namespace utils;

TEST(clFTensor, canCreateEmpty) {
  clFTensor t;
  EXPECT_EQ(t.getZ(), 0);
  EXPECT_EQ(t.getX(), 0);
  EXPECT_EQ(t.getY(), 0);
}

TEST(clFTensor, canCreate) {
  clFTensor t2(1, 2, 3);

  EXPECT_EQ(t2.getX(), 1);
  EXPECT_EQ(t2.getY(), 2);
  EXPECT_EQ(t2.getZ(), 3);
}

TEST(clFTensor, canFlatten) {
  clFTensor t;
  auto flat = t.flatten();
  EXPECT_EQ(flat.getX(), 0);
  EXPECT_EQ(flat.getY(), 0);
  EXPECT_EQ(flat.getZ(), 0);

  clFTensor t2(1, 2, 3);
  flat = t2.flatten();

  EXPECT_EQ(flat.getX(), 2);
  EXPECT_EQ(flat.getY(), 1);
  EXPECT_EQ(flat.getZ(), 3);
}

TEST(clFTensor, canDeepCopy) {
  clFTensor t(1, 2, 3);
  clFTensor t2 = t;

  EXPECT_EQ(t2.getX(), 1);
  EXPECT_EQ(t2.getY(), 2);
  EXPECT_EQ(t2.getZ(), 3);

  // The buffer should not have been copied
  EXPECT_NE(t2.getBuffer()(), t.getBuffer()());
}

TEST(clFTensor, canShallowCopy) {
  clFTensor t(1, 2, 3);
  clFTensor t2 = t.shallowCopy();

  EXPECT_EQ(t2.getX(), 1);
  EXPECT_EQ(t2.getY(), 2);
  EXPECT_EQ(t2.getZ(), 3);

  // The buffer should have been copied
  EXPECT_EQ(t2.getBuffer()(), t.getBuffer()());
}

TEST(clFTensor, canCreateSubmatrix) {
  FloatMatrix m(1, 2);
  m(0, 0) = 1;
  m(0, 1) = 1;

  clFTensor tensor(1, 2, 3);
  clFMatrix mat1 = tensor.getMatrix(0);

  EXPECT_EQ(mat1.getRows(), 1);
  EXPECT_EQ(mat1.getCols(), 2);
  mat1 = m;
  mat1.ipadd(1.0f, mat1);

  clFMatrix mat2 = tensor.getMatrix(1);
  EXPECT_EQ(mat2.getRows(), 1);
  EXPECT_EQ(mat2.getCols(), 2);
  mat2 = m;
  mat2.ipadd(2.0f, mat2);

  clFMatrix mat3 = tensor.getMatrix(2);
  EXPECT_EQ(mat3.getRows(), 1);
  EXPECT_EQ(mat3.getCols(), 2);
  mat3 = m;
  mat3.ipadd(5.0f, mat3);

  utils::cl_wrapper.getDefaultQueue().finish();

  auto entire_tensor = clFMatrix::fromSubbuffer(tensor.getBuffer(), 6, 1);
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
  auto subtensors = tensor.shallowSplit(3);

  for (size_t i = 0; i < 3; i++) {
    EXPECT_EQ(subtensors[i].getX(), 1);
    EXPECT_EQ(subtensors[i].getY(), 2);
    EXPECT_EQ(subtensors[i].getZ(), 1);
  }

  clFMatrix mat1 = subtensors[0].getMatrix(0);

  EXPECT_EQ(mat1.getRows(), 1);
  EXPECT_EQ(mat1.getCols(), 2);
  mat1 = m;
  mat1.ipadd(1.0f, mat1);

  clFMatrix mat2 = subtensors[1].getMatrix(0);
  EXPECT_EQ(mat2.getRows(), 1);
  EXPECT_EQ(mat2.getCols(), 2);
  mat2 = m;
  mat2.ipadd(2.0f, mat2);

  clFMatrix mat3 = subtensors[2].getMatrix(0);
  EXPECT_EQ(mat3.getRows(), 1);
  EXPECT_EQ(mat3.getCols(), 2);
  mat3 = m;
  mat3.ipadd(5.0f, mat3);

  utils::cl_wrapper.getDefaultQueue().finish();

  auto entire_tensor = clFMatrix::fromSubbuffer(tensor.getBuffer(), 6, 1);
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
