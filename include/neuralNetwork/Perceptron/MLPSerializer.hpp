
#pragma once
#include "tscl.hpp"

#include "MLPerceptron.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>

namespace nnet {

  /** @brief Utility class for serializing and deserializing MLPerceptrons
   *
   */
  class MLPSerializer {
  public:
    /**
     * @returns a new MLPerceptronSerializer, or nullptr on failure
     */
    virtual MLPerceptron readFromFile(const std::filesystem::path &path) = 0;

    /**
     * @returns a new MLPerceptronSerializer, or nullptr on failure
     */
    virtual MLPerceptron readFromStream(std::istream &stream) = 0;

    /**
     *
     * @param path Path to the file to write to
     * @param perceptron The perceptron to write
     * @return true on success, false on failure
     */
    virtual bool writeToFile(const std::filesystem::path &path,
                             const MLPerceptron &perceptron) = 0;

    /**
     *
     * @param stream The stream to write to
     * @param perceptron The perceptron to write
     * @return true on success, false on failure
     */
    virtual bool writeToStream(std::ostream &stream, const MLPerceptron &perceptron) = 0;
  };
}   // namespace nnet