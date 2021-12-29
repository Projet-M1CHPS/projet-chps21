#pragma once
#include "MLPSerializer.hpp"

namespace nnet {
  /** Utility class for serializing and deserializing MLPerceptrons
   *
   */
  class PlainTextMLPSerializer : public MLPSerializer {
  public:
    MLPerceptron<float> readFromFile(const std::filesystem::path &path) override;

    MLPerceptron<float> readFromStream(std::istream &stream) override;

    bool writeToFile(const std::filesystem::path &path,
                     const MLPerceptron<float> &perceptron) override;

    bool writeToStream(std::ostream &stream, const MLPerceptron<float> &perceptron) override;

  private:
    MLPTopology readTopology(std::istream &stream);
    void writeTopology(std::ostream &stream, const MLPTopology &topology);

    std::vector<af::ActivationFunctionType> readActivationFunctions(std::istream &stream);
    void writeActivationFunctions(std::ostream &stream,
                                  const std::vector<af::ActivationFunctionType> &functions);

    void readMatrices(std::istream &stream, std::vector<math::FloatMatrix> &matrices,
                      const std::string &section_name);

    void writeMatrices(std::ostream &stream, const std::vector<math::FloatMatrix> &matrices,
                       const std::string &section_name);
  };
}   // namespace nnet