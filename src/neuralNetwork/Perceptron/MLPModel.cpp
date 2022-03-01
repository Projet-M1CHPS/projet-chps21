
#include "MLPModel.hpp"
#include <MLPModelSerializer.hpp>
#include <utility>

namespace nnet {


  MLPModel::MLPModel(std::shared_ptr<utils::clWrapper> wrapper_ptr) : Model(wrapper_ptr) {
    perceptron = std::make_unique<MLPerceptron>(wrapper_ptr.get());
  }

  MLPModel::MLPModel(std::shared_ptr<utils::clWrapper> wrapper_ptr,
                     std::unique_ptr<MLPerceptron> &&perceptron)
      : Model(std::move(wrapper_ptr)) {
    this->perceptron = std::move(perceptron);
  }


  math::clFMatrix MLPModel::predict(math::clFMatrix const &input) const {
    return perceptron->predict(input);
  }

  std::unique_ptr<MLPModel> MLPModel::random(const std::shared_ptr<utils::clWrapper> &wrapper_ptr,
                                             MLPTopology const &topology,
                                             af::ActivationFunctionType af) {
    auto res = std::make_unique<MLPModel>(wrapper_ptr);
    auto &mlp = res->getPerceptron();
    mlp.setTopology(topology);
    mlp.setActivationFunction(af);
    mlp.randomizeWeight();

    return res;
  }

  std::unique_ptr<MLPModel>
  MLPModel::randomReluSigmoid(const std::shared_ptr<utils::clWrapper> &wrapper_ptr,
                              MLPTopology const &topology) {
    auto res = random(wrapper_ptr, topology, af::ActivationFunctionType::leakyRelu);
    auto &mlp = res->getPerceptron();

    for (size_t i = 0; i < topology.size() - 1; i += 2) {
      mlp.setActivationFunction(af::ActivationFunctionType::sigmoid, i);
    }
    // The last layer should be a sigmoid for the result to be in [0;1]
    mlp.setActivationFunction(af::ActivationFunctionType::sigmoid, topology.size() - 2);

    return res;
  }


  bool MLPModel::load(const std::filesystem::path &path) {
    try {
      auto tmp = MLPModelSerializer::readFromFile(path, cl_wrapper_ptr);
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

}   // namespace nnet