#pragma once

#include "Model.hpp"
#include "neuralNetwork/Perceptron/MLPerceptron.hpp"

namespace nnet {
  template<typename real = float, typename = std::enable_if<std::is_floating_point_v<real>>>
  class MLPModel : public Model<real> {
  public:
    MLPModel() { perceptron = std::make_unique<MLPerceptron<real>>(); }
    ~MLPModel() = default;

    math::Matrix<real> predict(math::Matrix<real> const &input) override {
      return perceptron->predict(input);
    }

    [[nodiscard]] MLPerceptron<real> &getPerceptron() { return *perceptron; }
    [[nodiscard]] MLPerceptron<real> const &getPerceptron() const { return *perceptron; }

  private:
    std::unique_ptr<MLPerceptron<real>> perceptron;
  };

  template<typename real = float, typename = std::enable_if<std::is_floating_point_v<real>>>
  class MLPModelFactory {
    using RModel = MLPModel<real>;

  public:
    MLPModelFactory() = delete;


    static std::unique_ptr<RModel> random(MLPTopology const &topology) {
      auto res = std::make_unique<RModel>();
      auto &mlp = res->getPerceptron();
      mlp.setTopology(topology);
      mlp.setActivationFunction(af::ActivationFunctionType::sigmoid);
      mlp.randomizeSynapses();
      return res;
    }

    static std::unique_ptr<RModel> randomSigReluAlt(MLPTopology const &topology) {
      auto res = std::make_unique<RModel>();
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