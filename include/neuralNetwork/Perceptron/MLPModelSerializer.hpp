#pragma once
#include <MLPModel.hpp>
#include <filesystem>
#include <iostream>

namespace nnet {

  /** @brief MLPModelSerializer specialization for storing MLPModel as a plain text file
   *
   */
  class MLPModelSerializer {
  public:
    MLPModelSerializer() = delete;

    /**
     * @returns a new MLPModelSerializer, or nullptr on failure
     */
    static MLPModel readFromFile(const std::filesystem::path &path);

    /**
     * @returns a new MLPModelSerializer, or nullptr on failure
     */
    static MLPModel readFromStream(std::istream &stream);

    /**
     * @param path Path to the file to write to
     * @param model The model to write
     * @return true on success, false on failure
     */
    static bool writeToFile(const std::filesystem::path &path, const MLPModel &model);

    /**
     *
     * @param stream The stream to write to
     * @param model The model to write
     * @return true on success, false on failure
     */
    static bool writeToStream(std::ostream &stream, const MLPModel &model);
  };
}   // namespace nnet