#include "controlSystem/inputSet.hpp"
#include <gtest/gtest.h>

using namespace control;

TEST(InputSetTest, CanBuildEmptySet) {
  InputSet<float> input_set;
  EXPECT_EQ(0, input_set.size());
}

TEST(InputSetTest, ReturnsSetSize) {
  InputSet<float> input_set;
  EXPECT_EQ(0, input_set.size());
  EXPECT_TRUE(input_set.empty());
  input_set.append("test", math::Matrix<float>());
  input_set.append("test", math::Matrix<float>());
  input_set.append("test", math::Matrix<float>());
  input_set.append("test", math::Matrix<float>());
  input_set.append("test", math::Matrix<float>());
  EXPECT_EQ(5, input_set.size());
  EXPECT_FALSE(input_set.empty());
}

TEST(InputSetTest, CanAppendToSet) {
  InputSet<float> input_set;

  input_set.append("test", math::Matrix<float>({1, 2, 3}));
  input_set.append("test2", math::Matrix<float>({3, 2}));
  EXPECT_EQ(2, input_set.size());
  EXPECT_STREQ("test", input_set.getPath(0).c_str());
  EXPECT_STREQ("test2", input_set.getPath(1).c_str());


  {
    auto &mat = input_set[0];
    EXPECT_EQ(1, mat.getCols());
    EXPECT_EQ(3, mat.getRows());
    EXPECT_EQ(1, mat(0, 0));
    EXPECT_EQ(2, mat(1, 0));
    EXPECT_EQ(3, mat(2, 0));
  }
  {
    auto &mat = input_set[1];
    EXPECT_EQ(1, mat.getCols());
    EXPECT_EQ(2, mat.getRows());
    EXPECT_EQ(3, mat(0, 0));
    EXPECT_EQ(2, mat(1, 0));
  }
}

TEST(InputSetTest, ThrowOnInvalidIndex) {
  InputSet<float> input_set;
  EXPECT_ANY_THROW(input_set[0]);

  input_set.append("", math::Matrix<float>());
  EXPECT_ANY_THROW(input_set[1]);
}

TEST(InputSetTest, CanUnload) {
  InputSet<float> input_set;
  input_set.append("test", math::Matrix<float>());
  input_set.unload();
  EXPECT_EQ(0, input_set.size());

  input_set = InputSet<float>();
  EXPECT_NO_THROW(input_set.unload());
}
