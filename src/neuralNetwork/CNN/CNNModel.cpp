#include "neuralNetwork/CNN/CNNModel.hpp"

namespace cnnet {

  CNNModel::CNNModel() {
    cnn = std::make_unique<CNN>();
    mlp = std::make_unique<MLPerceptron>();
  }


  math::clFMatrix CNNModel::predict(math::clFMatrix const &input) const {
    // TODO: Use clFMatrix everywhere
    auto tmp_input = input.toFloatMatrix(true);
    auto tmp_flatten = flatten.toFloatMatrix(true);
    cnn->predict(tmp_input, tmp_flatten);

    // TODO: Remove flatten object member and use a local variable instead
    clFMatrix tmp(tmp_flatten);

    return mlp->predict(tmp);
  }

  std::unique_ptr<CNNModel> CNNModel::random(CNNTopology const &topology, MLPTopology &mlp_topology) {
    auto res = std::make_unique<CNNModel>();

    auto &cnn = res->getCnn();
    cnn.setTopology(topology);
    // TODO: Implement random initialization
    // cnn.randomizeWeight();

    const size_t size = cnn.getOutputSize();
    res->flatten = clFMatrix(size, 1);
    mlp_topology.push_front(size);

    auto &mlp = res->getMlp();
    mlp.setTopology(mlp_topology);
    mlp.setActivationFunction(af::ActivationFunctionType::sigmoid);
    mlp.randomizeWeight();

    return res;
  }

}   // namespace cnnet
