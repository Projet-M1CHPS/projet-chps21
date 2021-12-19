#include "controllerParameters.hpp"
#include "gtest/gtest.h"

using namespace control;

TEST(ControllerParameters, CanBuild) {
  ControllerParameters p("i", "o", false);
  EXPECT_STREQ(p.getInputPath().c_str(), "i");
  EXPECT_STREQ(p.getOutputPath().c_str(), "o");
  EXPECT_FALSE(p.isVerbose());
}

TEST(ControllerParameters, CanSet) {
  ControllerParameters p("i", "o", false);

  p.setOutputPath("o2");
  p.setInputPath("i2");
  p.setVerbose(true);

  EXPECT_STREQ(p.getOutputPath().c_str(), "o2");
  EXPECT_STREQ(p.getInputPath().c_str(), "i2");
  EXPECT_TRUE(p.isVerbose());
}

TEST(ControllerParameters, CanCopy) {
  ControllerParameters p("i", "o", false);
  ControllerParameters p2(p);

  EXPECT_STREQ(p2.getInputPath().c_str(), "i");
  EXPECT_STREQ(p2.getOutputPath().c_str(), "o");
  EXPECT_FALSE(p2.isVerbose());

  // Test move copy
  ControllerParameters p3 = std::move(p);
  EXPECT_STREQ(p3.getInputPath().c_str(), "i");
  EXPECT_STREQ(p3.getOutputPath().c_str(), "o");
  EXPECT_FALSE(p3.isVerbose());

  EXPECT_STREQ(p.getOutputPath().c_str(), "");
  EXPECT_STREQ(p.getInputPath().c_str(), "");
  EXPECT_FALSE(p.isVerbose());
}

TEST(TrainingControllerParameters, CanBuild) {
  TrainingControllerParameters p("i", "o", 10, 15, false);
  EXPECT_STREQ(p.getInputPath().c_str(), "i");
  EXPECT_STREQ(p.getOutputPath().c_str(), "o");
  EXPECT_EQ(p.getMaxEpoch(), 10);
  EXPECT_EQ(p.getBatchSize(), 15);
}

TEST(TrainingControllerParameters, CanSet) {
  TrainingControllerParameters p("i", "o", 10, 15, false);

  p.setInputPath("i2");
  p.setOutputPath("o2");
  p.setMaxEpoch(20);
  p.setBatchSize(25);
  p.setVerbose(true);

  EXPECT_STREQ(p.getInputPath().c_str(), "i2");
  EXPECT_STREQ(p.getOutputPath().c_str(), "o2");
  EXPECT_EQ(p.getMaxEpoch(), 20);
  EXPECT_EQ(p.getBatchSize(), 25);
  EXPECT_TRUE(p.isVerbose());
}

TEST(TrainingControllerParameters, CanCopy) {
  TrainingControllerParameters p("i", "o", 10, 15, false);
  TrainingControllerParameters p2(p);

  EXPECT_STREQ(p2.getInputPath().c_str(), "i");
  EXPECT_STREQ(p2.getOutputPath().c_str(), "o");
  EXPECT_EQ(p2.getMaxEpoch(), 10);
  EXPECT_EQ(p2.getBatchSize(), 15);

  // Test Move copy
  TrainingControllerParameters p3 = std::move(p);
  EXPECT_STREQ(p3.getInputPath().c_str(), "i");
  EXPECT_STREQ(p3.getOutputPath().c_str(), "o");
  EXPECT_EQ(p3.getMaxEpoch(), 10);
  EXPECT_EQ(p3.getBatchSize(), 15);

  EXPECT_STREQ(p.getOutputPath().c_str(), "");
  EXPECT_STREQ(p.getInputPath().c_str(), "");
  EXPECT_EQ(p.getMaxEpoch(), 10);
  EXPECT_EQ(p.getBatchSize(), 15);
}