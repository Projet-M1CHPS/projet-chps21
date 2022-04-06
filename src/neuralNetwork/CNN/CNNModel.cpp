#include "neuralNetwork/CNN/CNNModel.hpp"

namespace nnet {

  CNNModel::CNNModel() {
    cnn = std::make_unique<CNN>();
    mlp = std::make_unique<MLPerceptron>();
  }


  math::clFMatrix CNNModel::predict(math::clFMatrix const &input) const {
    // TODO: Remove flatten object member and use a local variable instead
    math::clFMatrix _flatten = clFMatrix(cnn->getOutputSize(), 1);

    cnn->predict(input, _flatten);

    return mlp->predict(_flatten);
  }

  std::unique_ptr<CNNModel> CNNModel::random(CNNTopology const &topology,
                                             MLPTopology &mlp_topology) {
    auto res = std::make_unique<CNNModel>();

    auto &cnn = res->getCnn();
    cnn.setTopology(topology);
    // TODO: Implement random initialization
    // cnn.randomizeWeight();

    const size_t size = cnn.getOutputSize();
    res->flatten = clFMatrix(size, 1);
    mlp_topology.pushFront(size);

    auto &mlp = res->getMlp();
    mlp.setTopology(mlp_topology);
    mlp.setActivationFunction(af::ActivationFunctionType::sigmoid);
    mlp.randomizeWeight();

    return res;
  }

}   // namespace cnnet
