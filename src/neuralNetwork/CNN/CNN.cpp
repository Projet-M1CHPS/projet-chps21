#include "CNN.hpp"

namespace nnet {

  CNN::CNN(CNN &&other) noexcept {
    this->topology = std::move(other.topology);
    this->tree = std::move(other.tree);
  }

  CNN &CNN::operator=(CNN &&other) noexcept {
    this->topology = std::move(other.topology);
    this->tree = std::move(other.tree);
    return *this;
  }

  void CNN::setTopology(CNNTopology const &cnn_topology) {
    tree.build(cnn_topology);

    layers = topology.convertToLayer();

    this->topology = cnn_topology;
  }


  void CNN::randomizeWeight() { assert(false && "Not implemented"); }


  void CNN::predict(clFTensor const &input, clFTensor &output) {
    if (layers.empty()) { throw std::runtime_error("no layer in cnn"); }

    clFTensor output_tensor;

    output_tensor = tree.getRoot()->getLayer()->compute(input);
    for (auto &layer : layers) { output_tensor = layer->compute(output_tensor); }

    utils::cl_wrapper.getDefaultQueue().finish();

    output = clFTensor(output_tensor);
  }

}   // namespace nnet