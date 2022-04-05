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

TEST(clFTensor, canGetDimensions) {
  clFTensor t(1, 2, 3);

  EXPECT_EQ(t.getX(), 1);
  EXPECT_EQ(t.getY(), 2);
  EXPECT_EQ(t.getZ(), 3);
}

TEST(clFTensor, canDeepCopy) {
  clFTensor t(1, 2, 3);
  clFTensor t2 = t;

  EXPECT_EQ(t2.getX(), 1);
  EXPECT_EQ(t2.getY(), 2);
  EXPECT_EQ(t2.getZ(), 3);
}

TEST(clFTensor, canMoveCopy) {
  clFTensor t(1, 2, 3);
  auto old_buffer = t.getBuffer();
  clFTensor t2 = std::move(t);

  EXPECT_EQ(old_buffer, t2.getBuffer());

  EXPECT_EQ(t2.getX(), 1);
  EXPECT_EQ(t2.getY(), 2);
  EXPECT_EQ(t2.getZ(), 3);
}

TEST(clFTensor, canMoveCopy) {
  clFTensor t(1, 2, 3);
  auto old_buffer = t.getBuffer();
  clFTensor t2 = std::move(t);

  EXPECT_EQ(old_buffer, t2.getBuffer());

  EXPECT_EQ(t2.getX(), 1);
  EXPECT_EQ(t2.getY(), 2);
  EXPECT_EQ(t2.getZ(), 3);
}