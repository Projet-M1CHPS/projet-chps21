#include "classifierClass.hpp"
#include <gtest/gtest.h>


using namespace control::classifier;

TEST(ClassLabelTest, CanBuild) {
  ClassLabel label(0, "label");
  EXPECT_EQ(0, label.getId());
  EXPECT_STREQ("label", label.getName().c_str());
}

TEST(ClassLabelTest, CanSet) {
  ClassLabel label(0, "label");
  label.setId(1);
  label.setName("newLabel");

  EXPECT_EQ(1, label.getId());
  EXPECT_STREQ("newLabel", label.getName().c_str());
}

TEST(ClassLabelTest, CanCopy) {
  ClassLabel label(0, "label");
  ClassLabel label2(label);

  EXPECT_EQ(0, label2.getId());
  EXPECT_STREQ("label", label2.getName().c_str());

  // Test move copy
  ClassLabel label3 = std::move(label);
  EXPECT_EQ(0, label3.getId());
  EXPECT_STREQ("label", label3.getName().c_str());

  EXPECT_EQ(0, label.getId());
  EXPECT_STREQ("", label.getName().c_str());
}

TEST(ClassLabelTest, CanCompare) {
  ClassLabel label1(0, "label");
  ClassLabel label2(0, "label");
  ClassLabel label3(1, "label");
  ClassLabel label4(0, "label2");

  // Equality comparisons
  EXPECT_TRUE(label1 == label2);
  EXPECT_FALSE(label1 == label3);
  EXPECT_FALSE(label1 == label4);
  EXPECT_TRUE(label1 != label3);
  EXPECT_TRUE(label1 != label4);

  // Less than comparisons
  EXPECT_TRUE(label1 < label3);
  EXPECT_FALSE(label1 < label4);
  EXPECT_FALSE(label3 < label1);
  EXPECT_FALSE(label4 < label1);

  // Greater than comparisons
  EXPECT_FALSE(label1 > label3);
  EXPECT_FALSE(label1 > label4);
  EXPECT_TRUE(label3 > label1);
  EXPECT_FALSE(label4 > label1);
}

// This is useful for serialization
TEST(ClassLabelTest, CanStream) {
  ClassLabel label(0, "label");
  std::stringstream ss;
  ss << label;
  EXPECT_STREQ("Class 0: label", ss.str().c_str());
}

TEST(CClassLabelSetTest, CanBuild) {
  CClassLabelSet list;
  EXPECT_EQ(0, list.size());
  EXPECT_TRUE(list.empty());
  EXPECT_TRUE(list.begin() == list.end());
}

TEST(CClassLabelSetTest, CanAppend) {
  CClassLabelSet list;
  list.append(ClassLabel(0, "label"));
  EXPECT_EQ(1, list.size());
  EXPECT_FALSE(list.empty());
  EXPECT_FALSE(list.begin() == list.end());
  auto label = list[0];

  EXPECT_EQ(0, label.getId());
  EXPECT_STREQ("label", label.getName().c_str());
  label = list.begin()->second;
  EXPECT_EQ(0, label.getId());
  EXPECT_STREQ("label", label.getName().c_str());
}

TEST(CClassLabelSet, ThrowsOnInvalidAppend) {
  CClassLabelSet list;
  list.append(ClassLabel(0, "label"));
  EXPECT_THROW(list.append(ClassLabel(0, "label")), std::runtime_error);
}

TEST(CClassLabelSet, ThrowsOnInvalidAccess) {
  CClassLabelSet list;
  EXPECT_THROW(list[0], std::runtime_error);
  list.append(ClassLabel(0, "label"));
  EXPECT_THROW(list[1], std::runtime_error);
}