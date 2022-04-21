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
    cnn.randomizeWeight();

    const size_t size = cnn.getTopology().getCNNOutputSize();
    mlp_topology.pushFront(size);

    auto &mlp = res->getMlp();
    mlp.setTopology(mlp_topology);
    // TODO : check comment on init la fonction d activation
    mlp.setActivationFunction(af::ActivationFunctionType::sigmoid);
    mlp.randomizeWeight();

    return res;
  }

  math::clFTensor CNNModel::predict(cl::CommandQueue &queue, math::clFTensor const &inputs) const {
    math::clFTensor flatten = cnn->predict(queue, inputs);
    // TODO : C est quoi ca ???
    // return mlp->predict(flatten);

    return {1, 1, 1};
  }

  math::clFMatrix CNNModel::predict(cl::CommandQueue &queue, math::clFMatrix const &input) const {
    math::clFTensor buffer(input.getRows(), input.getCols(), 1);
    buffer[0].copy(input, queue, true);

    return cnn->predict(queue, buffer)[0];
  }

}   // namespace nnet
