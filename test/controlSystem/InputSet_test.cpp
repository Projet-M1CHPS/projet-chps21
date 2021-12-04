#include "controlSystem/inputSet.hpp"
#include <gtest/gtest.h>

using namespace control;

TEST(InputSetTest, CanBuildEmptySet) {
  InputSet input_set;
  EXPECT_EQ(0, input_set.size());
}

TEST(InputSetTest, ReturnsSetSize) {
  InputSet input_set;
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
  InputSet input_set;

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
  InputSet input_set;
  EXPECT_ANY_THROW(input_set[0]);

  input_set.append("", math::Matrix<float>());
  EXPECT_ANY_THROW(input_set[1]);
}

TEST(InputSetTest, CanUnload) {
  InputSet input_set;
  input_set.append("test", math::Matrix<float>());
  input_set.unload();
  EXPECT_EQ(0, input_set.size());

  input_set = InputSet();
  EXPECT_NO_THROW(input_set.unload());
}

TEST(TrainingSetTest, CanCreateEmpty) {
  TrainingSet training_set;
  EXPECT_EQ(0, training_set.size());
}

TEST(TrainingSetTest, CanAppendToEval) {
  TrainingSet training_set;
  training_set.appendToEvalSet("test", 2, math::Matrix<float>({1, 2, 3}));
  training_set.appendToEvalSet("test2", 1, math::Matrix<float>({3, 2}));
  EXPECT_EQ(2, training_set.size());

  EXPECT_STREQ("test", training_set.getEvalPath(0).c_str());
  EXPECT_STREQ("test2", training_set.getEvalPath(1).c_str());

  {
    auto &mat = training_set.getEvalMat(0);
    EXPECT_EQ(1, mat.getCols());
    EXPECT_EQ(3, mat.getRows());
    EXPECT_EQ(1, mat(0, 0));
    EXPECT_EQ(2, mat(1, 0));
    EXPECT_EQ(3, mat(2, 0));
  }
  {
    auto &mat = training_set.getEvalMat(1);
    EXPECT_EQ(1, mat.getCols());
    EXPECT_EQ(2, mat.getRows());
    EXPECT_EQ(3, mat(0, 0));
    EXPECT_EQ(2, mat(1, 0));
  }
}

TEST(TrainingSetTest, CanAppendToTraining) {
  TrainingSet training_set;
  training_set.appendToTrainingSet("test", 2, math::Matrix<float>({1, 2, 3}));
  training_set.appendToTrainingSet("test2", 1, math::Matrix<float>({3, 2}));
  EXPECT_EQ(2, training_set.size());

  EXPECT_STREQ("test", training_set.getTrainingPath(0).c_str());
  EXPECT_STREQ("test2", training_set.getTrainingPath(1).c_str());

  {
    auto &mat = training_set.getTrainingMat(0);
    EXPECT_EQ(1, mat.getCols());
    EXPECT_EQ(3, mat.getRows());
    EXPECT_EQ(1, mat(0, 0));
    EXPECT_EQ(2, mat(1, 0));
    EXPECT_EQ(3, mat(2, 0));
  }
  {
    auto &mat = training_set.getTrainingMat(1);
    EXPECT_EQ(1, mat.getCols());
    EXPECT_EQ(2, mat.getRows());
    EXPECT_EQ(3, mat(0, 0));
    EXPECT_EQ(2, mat(1, 0));
  }
}

TEST(TrainingSetTest, CanReturnSize) {
  TrainingSet training_set;
  EXPECT_EQ(0, training_set.trainingSetSize());
  EXPECT_EQ(0, training_set.evalSetSize());
  EXPECT_TRUE(training_set.empty());

  training_set.appendToTrainingSet("test", 1, math::Matrix<float>());
  training_set.appendToTrainingSet("test", 1, math::Matrix<float>());
  training_set.appendToTrainingSet("test", 1, math::Matrix<float>());

  EXPECT_EQ(3, training_set.trainingSetSize());
  EXPECT_EQ(0, training_set.evalSetSize());
  EXPECT_FALSE(training_set.empty());

  training_set.appendToEvalSet("test", 1, math::Matrix<float>());
  training_set.appendToEvalSet("test", 1, math::Matrix<float>());
  training_set.appendToEvalSet("test", 1, math::Matrix<float>());

  EXPECT_EQ(3, training_set.trainingSetSize());
  EXPECT_EQ(3, training_set.evalSetSize());
  EXPECT_FALSE(training_set.empty());
}

TEST(TrainingSetTest, CanSetCategories) {
  TrainingSet training_set;
  auto categories = {"0", "1", "2"};
  training_set.setCategories(categories.begin(), categories.end());
  EXPECT_EQ(3, training_set.getCategoryCount());
  auto &categories2 = training_set.getCategories();

  EXPECT_TRUE(std::equal(categories.begin(), categories.end(), categories2.begin()));
}

TEST(TrainingSetTest, ThrowOnInvalidIndex) {
  TrainingSet training_set;

  EXPECT_ANY_THROW(training_set.getCategory(0));
  EXPECT_ANY_THROW(training_set.getEvalMat(0));
  EXPECT_ANY_THROW(training_set.getEvalPath(0));
  EXPECT_ANY_THROW(training_set.getTrainingMat(0));
  EXPECT_ANY_THROW(training_set.getTrainingPath(0));
}

TEST(TrainingSetTest, CanUnload) {
  TrainingSet training_set;

  EXPECT_NO_THROW(training_set.unload());
  training_set.appendToTrainingSet("test", 1, math::Matrix<float>());
  training_set.appendToEvalSet("test", 1, math::Matrix<float>());
  auto categories = {"0", "1", "2"};
  training_set.setCategories(categories.begin(), categories.end());

  EXPECT_NO_THROW(training_set.unload());
  EXPECT_TRUE(training_set.empty());
}