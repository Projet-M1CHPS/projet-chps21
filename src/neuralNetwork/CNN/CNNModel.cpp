#include "neuralNetwork/CNN/CNNModel.hpp"

namespace cnnet {

  CNNModel::CNNModel(std::shared_ptr<utils::clWrapper> wrapper_ptr) : Model(wrapper_ptr) {
    cnn = std::make_unique<CNN>();
    mlp = std::make_unique<MLPerceptron>(wrapper_ptr.get());
  }


  math::clFMatrix CNNModel::predict(math::clFMatrix const &input) const {
    // TODO: Use clFMatrix everywhere
    auto tmp_input = input.toFloatMatrix(*cl_wrapper_ptr, true);
    auto tmp_flatten = flatten.toFloatMatrix(*cl_wrapper_ptr, true);
    cnn->predict(tmp_input, tmp_flatten);

    // TODO: Remove flatten object member and use a local variable instead
    clFMatrix tmp(tmp_flatten, *cl_wrapper_ptr);

    return mlp->predict(tmp);
  }

  std::unique_ptr<CNNModel> CNNModel::random(CNNTopology const &topology, MLPTopology &mlp_topology,
                                             std::shared_ptr<utils::clWrapper> wrapper_ptr) {
    auto res = std::make_unique<CNNModel>(wrapper_ptr);

    auto &cnn = res->getCnn();
    cnn.setTopology(topology);
    // TODO: Implement random initialization
    // cnn.randomizeWeight();

    const size_t size = cnn.getOutputSize();
    res->flatten = clFMatrix(size, 1, *wrapper_ptr);
    mlp_topology.push_front(size);

    auto &mlp = res->getMlp();
    mlp.setTopology(mlp_topology);
    mlp.setActivationFunction(af::ActivationFunctionType::sigmoid);
    mlp.randomizeWeight();

    return res;
  }

}   // namespace cnnet
