
#include "MLPModel.hpp"
#include <MLPModelSerializer.hpp>

namespace nnet {


  math::FloatMatrix MLPModel::predict(math::FloatMatrix const &input) const {
    return perceptron->predict(input);
  }

  bool MLPModel::load(const std::filesystem::path &path) {
    try {
      auto tmp = MLPModelSerializer::readFromFile(path);
      *this = std::move(tmp);
      return true;
    } catch (std::exception &e) { return false; }
  }

  bool MLPModel::save(const std::filesystem::path &path) const {
    try {
      MLPModelSerializer::writeToFile(path, *this);
      return true;
    } catch (std::exception &e) { return false; }
  }

  MLPModel::MLPModel() { perceptron = std::make_unique<MLPerceptron>(); }

  MLPModel::MLPModel(std::unique_ptr<MLPerceptron> &&perceptron) {
    this->perceptron = std::move(perceptron);
  }

  std::unique_ptr<MLPModel> MLPModel::random(MLPTopology const &topology,
                                             af::ActivationFunctionType af) {
    auto res = std::make_unique<MLPModel>();
    auto &mlp = res->getPerceptron();
    mlp.setTopology(topology);
    mlp.setActivationFunction(af);
    mlp.randomizeWeight()

            return res;
  }

  std::unique_ptr<MLPModel> MLPModel::randomReluSigmoid(MLPTopology const &topology) {
    auto res = random(topology, af::ActivationFunctionType::leakyRelu);
    auto &mlp = res->getPerceptron();

    for (size_t i = 0; i < topology.size() - 1; i += 2) {
      mlp.setActivationFunction(af::ActivationFunctionType::sigmoid, i);
    }
    // The last layer should be a sigmoid for the result to be in [0;1]
    mlp.setActivationFunction(af::ActivationFunctionType::sigmoid, topology.size() - 1);

    return res;
  }

}   // namespace nnet