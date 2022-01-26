#pragma once
#include "MLPModelSerializer.hpp"

namespace nnet {

  /** @brief MLPModelSerializer specialization for storing MLPModel as a plain text file
   *
   */
  class PlainTextMLPModelSerializer final : public MLPModelSerializer {
  public:
    /**
     * @returns a new MLPModelSerializer, or nullptr on failure
     */
    MLPModel readFromFile(const std::filesystem::path &path) override;

    /**
     * @returns a new MLPModelSerializer, or nullptr on failure
     */
    MLPModel readFromStream(std::istream &stream) override;

    /**
     *
     * @param path Path to the file to write to
     * @param model The model to write
     * @return true on success, false on failure
     */
    bool writeToFile(const std::filesystem::path &path, const MLPModel &model) override;

    /**
     *
     * @param stream The stream to write to
     * @param model The model to write
     * @return true on success, false on failure
     */
    bool writeToStream(std::ostream &stream, const MLPModel &model) override;
  };
}   // namespace nnet