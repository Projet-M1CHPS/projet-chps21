#pragma once
#include "MLPerceptron.hpp"

namespace nnet {
  /** Utility class for serializing and deserializing MLPerceptrons
   *
   */
  class MLPSerializer {
  public:
    static MLPerceptron readFromFile(const std::filesystem::path &path);

    static MLPerceptron readFromStream(std::istream &stream);

    static bool writeToFile(const std::filesystem::path &path, const MLPerceptron &perceptron);

    static bool writeToStream(std::ostream &stream, const MLPerceptron &perceptron);
  };
}   // namespace nnet