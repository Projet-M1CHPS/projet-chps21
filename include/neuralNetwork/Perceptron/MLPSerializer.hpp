
#pragma once
#include "tscl.hpp"

#include "MLPModel.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>

namespace nnet {

  /** Utility class for serializing and deserializing MLPModels
   *
   *
   */
  class MLPModelSerializer {
  public:
    static std::unique_ptr<MLPModel<float>> readFromFile(const std::filesystem::path &path);
    static std::unique_ptr<MLPModel<float>> readFromStream(std::istream &stream);

    static bool writeToFile(const std::filesystem::path &path, const MLPModel<float> &model);
    static bool writeToStream(std::ostream &stream, const MLPModel<float> &model);
  };

  class MLPerceptronSerializer {
  public:
    static std::unique_ptr<MLPerceptron<float>> readFromFile(const std::filesystem::path &path);
    static std::unique_ptr<MLPerceptron<float>> readFromStream(std::istream &stream);

    static bool writeToFile(const std::filesystem::path &path,
                            const MLPerceptron<float> &perceptron);
    static bool writeToStream(std::ostream &stream, const MLPerceptron<float> &perceptron);
  };

}   // namespace nnet