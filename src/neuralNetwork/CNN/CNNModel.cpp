#include "neuralNetwork/CNN/CNNModel.hpp"

namespace nnet {

  CNNModel::CNNModel() {
    cnn = std::make_unique<CNN>();
    mlp = std::make_unique<MLPerceptron>();
  }


  std::unique_ptr<CNNModel> CNNModel::random(CNNTopology const &topology,
                                             MLPTopology &mlp_topology) {
    auto res = std::make_unique<CNNModel>();

    auto &cnn = res->getCnn();
    cnn.setTopology(topology);
    // TODO: Implement random initialization
    // cnn.randomizeWeight();

    const size_t size = cnn.getTopology().getCNNOutputSize();
    mlp_topology.pushFront(size);

    auto &mlp = res->getMlp();
    mlp.setTopology(mlp_topology);
    mlp.setActivationFunction(af::ActivationFunctionType::sigmoid);
    mlp.randomizeWeight();

    return res;
  }

  clFTensor CNNModel::predict(clFTensor const &inputs) const {
    clFTensor flatten = cnn->predict(inputs);

    // return mlp->predict(flatten);

    return clFTensor(1, 1, 1);
  }

  clFMatrix CNNModel::predict(clFMatrix const &input) const {
    clFTensor buffer(input.getRows(), input.getCols(), 1);
    buffer[0].copy(input, true);

    return cnn->predict(buffer)[0];
  }

}   // namespace nnet
