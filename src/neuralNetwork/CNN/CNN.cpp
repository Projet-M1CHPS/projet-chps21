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

    this->topology = cnn_topology;
  }


  void CNN::randomizeWeight() { assert(false && "Not implemented"); }


  void CNN::predict(math::clFMatrix const &input, math::clFMatrix &output) {
    // TODO : Implement this

    if (not tree.getRoot()) { throw std::runtime_error("Root node is not set"); }

    std::stack<std::shared_ptr<CNNNode>> stack;

    tree.getRoot()->getLayer()->compute(input);
    for (auto &i : tree.getRoot()->getChildren()) { stack.push(std::make_shared<CNNNode>(i)); }

    while (not stack.empty()) {
      std::shared_ptr<CNNNode> node = stack.top();
      stack.pop();

      node->getLayer()->compute(node->getFather()->getLayer()->getOutput(node->getId()));
      for (auto &i : node->getChildren()) { stack.push(std::make_shared<CNNNode>(i)); }
    }
    utils::cl_wrapper.getDefaultQueue().finish();
  }

}   // namespace nnet