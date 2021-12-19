
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
    /**
     * @returns a new MLPModelSerializer, or nullptr on failure
     */
    static std::unique_ptr<MLPModel<float>> readFromFile(const std::filesystem::path &path);

    /**
     * @returns a new MLPModelSerializer, or nullptr on failure
     */
    static std::unique_ptr<MLPModel<float>> readFromStream(std::istream &stream);

    /**
     *
     * @param path Path to the file to write to
     * @param model The model to write
     * @return true on success, false on failure
     */
    static bool writeToFile(const std::filesystem::path &path, const MLPModel<float> &model);

    /**
     *
     * @param stream The stream to write to
     * @param model The model to write
     * @return true on success, false on failure
     */
    static bool writeToStream(std::ostream &stream, const MLPModel<float> &model);
  };

  /** Utility class for serializing and deserializing MLPerceptrons
   *
   */
  class MLPerceptronSerializer {
  public:
    /**
     * @returns a new MLPerceptronSerializer, or nullptr on failure
     */
    static std::unique_ptr<MLPerceptron<float>> readFromFile(const std::filesystem::path &path);

    /**
     * @returns a new MLPerceptronSerializer, or nullptr on failure
     */
    static std::unique_ptr<MLPerceptron<float>> readFromStream(std::istream &stream);

    /**
     *
     * @param path Path to the file to write to
     * @param perceptron The perceptron to write
     * @return true on success, false on failure
     */
    static bool writeToFile(const std::filesystem::path &path,
                            const MLPerceptron<float> &perceptron);

    /**
     *
     * @param stream The stream to write to
     * @param perceptron The perceptron to write
     * @return true on success, false on failure
     */
    static bool writeToStream(std::ostream &stream, const MLPerceptron<float> &perceptron);
  };

}   // namespace nnet