

#include "classifierCollection.hpp"
#include "gtest/gtest.h"


using namespace control::classifier;

TEST(ClassifierInputSet, CanBuildEmptySet) {
  ClassifierInputSet input_set;
  EXPECT_EQ(0, input_set.size());
}

TEST(ClassifierInputSet, CanAppend) {
  auto map = std::make_shared<std::vector<ClassLabel>>();
  map->emplace_back(1, "Test");

  ClassifierInputSet input_set(map);
  input_set.append("test", math::Matrix<float>({1, 2, 3}));

  EXPECT_EQ(1, input_set.size());
  EXPECT_STREQ("test", input_set.getPath(0).c_str());

  {
    auto &mat = input_set[0];
    EXPECT_EQ(1, mat.getCols());
    EXPECT_EQ(3, mat.getRows());
    EXPECT_EQ(1, mat(0, 0));
    EXPECT_EQ(2, mat(1, 0));
    EXPECT_EQ(3, mat(2, 0));
  }

  auto label = input_set.getLabel(0);
  EXPECT_TRUE(label == ClassLabel::unknown);

  input_set.append("test2", &map->at(0), math::Matrix<float>({2, 3}));
  EXPECT_EQ(2, input_set.size());
  EXPECT_STREQ("test2", input_set.getPath(1).c_str());

  {
    auto &mat = input_set[1];
    EXPECT_EQ(1, mat.getCols());
    EXPECT_EQ(2, mat.getRows());
    EXPECT_EQ(2, mat(0, 0));
    EXPECT_EQ(3, mat(1, 0));
  }

  label = input_set.getLabel(1);
  EXPECT_TRUE(label.getId() == 1);
  EXPECT_STREQ("Test", label.getName().c_str());
}

TEST(ClassifierInputSet, ThrowOnInvalidIndex) {
  ClassifierInputSet training_set;

  EXPECT_ANY_THROW(training_set.getLabel(0));
}

TEST(ClassifierInputSet, CanShuffle) {
  auto map = std::make_shared<std::vector<ClassLabel>>();
  for (size_t i = 0; i < 10; i++) map->emplace_back(i, "Test");


  ClassifierInputSet training_set(map);

  std::vector<ClassLabel *> indexes(10);

  for (size_t i = 0; i < 10; i++) {
    training_set.append(std::to_string(i), &map->at(i), math::Matrix<float>({(float) i}));
    indexes[i] = &map->at(i);
  }
  training_set.shuffle(std::random_device()());

  EXPECT_TRUE(
          std::is_permutation(indexes.begin(), indexes.end(), training_set.getLabels().begin()));

  for (size_t i = 0; i < 10; i++) {
    EXPECT_EQ(training_set.getLabel(i).getId(), training_set[i](0, 0));
    EXPECT_EQ(training_set[i](0, 0), std::stoi(training_set.getPath(i)));
  }
}

TEST(ClassifierInputSet, CanUnload) {
  ClassifierInputSet training_set;

  EXPECT_NO_THROW(training_set.unload());
  training_set.append("test", math::Matrix<float>());
  training_set.append("test", math::Matrix<float>());

  EXPECT_NO_THROW(training_set.unload());
  EXPECT_TRUE(training_set.empty());
  EXPECT_TRUE(training_set.getLabels().empty());
}
