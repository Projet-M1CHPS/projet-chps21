
#include "MLPModel.hpp"

namespace nnet {

  MLPModel::MLPModel() { perceptron = std::make_unique<MLPerceptron>(); }


  std::unique_ptr<MLPModel> MLPModelFactory::random(MLPTopology const &topology) {
    auto res = std::make_unique<MLPModel>();
    auto &mlp = res->getPerceptron();
    mlp.setTopology(topology);
    mlp.setActivationFunction(af::ActivationFunctionType::sigmoid);
    mlp.randomizeWeight();
    return res;
  }

  std::unique_ptr<MLPModel>
  MLPModelFactory::randomSigReluAlt(MLPTopology const &topology) {
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

}   // namespace nnet