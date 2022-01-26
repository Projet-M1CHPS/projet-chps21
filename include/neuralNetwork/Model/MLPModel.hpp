#pragma once

#include "Model.hpp"
#include "neuralNetwork/Perceptron/MLPerceptron.hpp"

namespace nnet {

  class MLPModel : public Model {
  public:
    MLPModel();
    ~MLPModel() = default;

    MLPModel(const MLPModel &) = delete;
    MLPModel(MLPModel &&) noexcept = default;

    math::FloatMatrix predict(math::FloatMatrix const &input) override {
      return perceptron->predict(input);
    }

    [[nodiscard]] MLPerceptron &getPerceptron() { return *perceptron; }
    [[nodiscard]] MLPerceptron const &getPerceptron() const { return *perceptron; }

  private:
    std::unique_ptr<MLPerceptron> perceptron;
  };

  class MLPModelFactory {
  public:
    MLPModelFactory() = delete;

    static std::unique_ptr<MLPModel> random(MLPTopology const &topology);

    static std::unique_ptr<MLPModel> randomSigReluAlt(MLPTopology const &topology);
  };
}   // namespace nnet