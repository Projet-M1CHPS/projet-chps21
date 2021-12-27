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
  // Append with move copy of the matrix
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
  // Append without moving the matrix
  set.append(1, &label2, mat2);
  auto &mat3 = set[1];
  EXPECT_EQ(3, mat3.getRows());
  EXPECT_EQ(1, mat3.getCols());
  EXPECT_EQ(4, mat3(0, 0));
}

TEST(ClassifierTrainingSet, CanClear) {
  ClassifierTrainingSet set;
  ClassLabel label1(1, "label1");
  set.append(0, &label1, {1, 2, 3});
  set.clear();

  EXPECT_TRUE(set.empty());
}

TEST(ClassifierTrainingSet, CanShuffle) {
  ClassifierTrainingSet set;

  ClassLabel label1(1, "1");
  ClassLabel label2(2, "2");
  ClassLabel label3(3, "3");

  set.append(0, &label1, {1});
  set.append(1, &label2, {2});
  set.append(2, &label3, {3});

  set.shuffle(5);

  // Check we haven't lost any data
  EXPECT_EQ(3, set.size());

  std::vector<ClassLabel> shuffled_labels;
  std::vector<ClassLabel> labels = {label1, label2, label3};
  for (size_t i = 0; i < set.size(); i++) { shuffled_labels.push_back(set.getLabel(i)); }

  // Check if the labels are shuffled or not
  ASSERT_TRUE(std::is_permutation(labels.begin(), labels.end(), shuffled_labels.begin()));

  for (size_t i = 0; i < set.size(); i++) { EXPECT_EQ(set[i](0, 0), set.getLabel(i).getId()); }
}

TEST(ClassifierTrainingSet, ThrowsOnInvalidAppend) {
  ClassifierTrainingSet set;
  // Should throw if the label is null
  EXPECT_THROW(set.append(0, nullptr, {1, 2, 3}), std::invalid_argument);
  math::Matrix<float> mat({4, 5, 6});
  EXPECT_THROW(set.append(0, nullptr, mat), std::invalid_argument);
}
