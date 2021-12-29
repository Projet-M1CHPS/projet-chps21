#pragma once
#include "MLPModelSerializer.hpp"

namespace nnet {

  class PlainTextMLPModelSerializer final : public MLPModelSerializer {
  public:
    /**
     * @returns a new MLPModelSerializer, or nullptr on failure
     */
    MLPModel<float> readFromFile(const std::filesystem::path &path) override;

    /**
     * @returns a new MLPModelSerializer, or nullptr on failure
     */
    MLPModel<float> readFromStream(std::istream &stream) override;

    /**
     *
     * @param path Path to the file to write to
     * @param model The model to write
     * @return true on success, false on failure
     */
    bool writeToFile(const std::filesystem::path &path, const MLPModel<float> &model) override;

    /**
     *
     * @param stream The stream to write to
     * @param model The model to write
     * @return true on success, false on failure
     */
    bool writeToStream(std::ostream &stream, const MLPModel<float> &model) override;
  };
}   // namespace nnet