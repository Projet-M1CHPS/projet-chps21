#include "classifierInputSet.hpp"
#include <gtest/gtest.h>

using namespace control::classifier;


TEST(ClassifierTrainingSet, CanBuild) {
  ClassifierTrainingSet set;
  EXPECT_EQ(0, set.size());
  EXPECT_TRUE(set.empty());
  EXPECT_EQ(set.begin(), set.end());
}

TEST(ClassifierTrainingSet, CanAppend) {
  ClassifierTrainingSet set;
  ClassLabel label1(1, "label1");
  set.append(0, &label1, {1, 2, 3});

  EXPECT_EQ(1, set.size());
  EXPECT_FALSE(set.empty());
  EXPECT_NE(set.begin(), set.end());

  // Check the matrix is rightly copied
  auto &mat = set[0];
  EXPECT_EQ(3, mat.getRows());
  EXPECT_EQ(1, mat.getCols());
  EXPECT_EQ(1, mat(0, 0));

  // Check the label is rightly copied
  EXPECT_EQ(label1, set.getLabel(0));

  // Test we can Move Append
  ClassLabel label2(2, "label2");
  math::Matrix<float> mat2({4, 5, 6});
  auto ptr = mat2.getData();
  set.append(1, &label2, std::move(mat2));
  auto &mat3 = set[1];
  EXPECT_EQ(3, mat3.getRows());
  EXPECT_EQ(1, mat3.getCols());
  EXPECT_EQ(4, mat3(0, 0));
  EXPECT_EQ(ptr, mat3.getData());
}

TEST(ClassifierTrainingSet, ThrowsOnInvalidAppend) {
  ClassifierTrainingSet set;
  // Should throw if the label is null
  EXPECT_THROW(set.append(0, nullptr, {1, 2, 3}), std::invalid_argument);
}
