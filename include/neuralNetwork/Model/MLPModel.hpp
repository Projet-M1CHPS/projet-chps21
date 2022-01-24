#pragma once

#include "Model.hpp"
#include "neuralNetwork/Perceptron/MLPerceptron.hpp"

namespace nnet {
  class MLPModel : public Model {
  public:
    MLPModel() { perceptron = std::make_unique<MLPerceptron>(); }
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

    static std::unique_ptr<MLPModel> random(MLPTopology const &topology) {
      auto res = std::make_unique<MLPModel>();
      auto &mlp = res->getPerceptron();
      mlp.setTopology(topology);
      mlp.setActivationFunction(af::ActivationFunctionType::sigmoid);
      mlp.randomizeWeight();
      return res;
    }

    static std::unique_ptr<MLPModel> randomSigReluAlt(MLPTopology const &topology) {
      auto res = std::make_unique<MLPModel>();
      auto &mlp = res->getPerceptron();
      mlp.setTopology(topology);
      mlp.randomizeWeight();
      mlp.setActivationFunction(af::ActivationFunctionType::leakyRelu);

      for (size_t i = 0; i < topology.size() - 1; i++) {
        if (i % 2 == 0 or i == topology.size() - 1) {
          mlp.setActivationFunction(af::ActivationFunctionType::sigmoid, i);
        }
      }

      return res;
    }
  };
}   // namespace nnet