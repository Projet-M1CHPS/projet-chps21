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

    std::vector<clFMatrix> output_tensor(topology.getDepth());

    if (not tree.getRoot()) { throw std::runtime_error("Root node is not set"); }

    std::stack<CNNNode *> stack;

    output_tensor[0] = tree.getRoot()->getLayer()->compute(input);
    for (auto &i : tree.getRoot()->getChildren()) { stack.push(&i); }

    while (not stack.empty()) {
      CNNNode *node = stack.top();
      stack.pop();

      // TODO : recuperer le node->getLocalId() eme element du future tensor
      std::cout << "call" << std::endl;
      output_tensor[node->getGlobalId()] = node->getLayer()->compute(output_tensor[node->getFather()->getGlobalId()]);
      for (auto &i : node->getChildren()) { stack.push(&i); }
    }
    utils::cl_wrapper.getDefaultQueue().finish();


    // TODO : Remove this
    FloatMatrix tmp_out = output.toFloatMatrix(true);
    size_t index = 0;
    for(auto &leave : tree.getLeaves())
    {
      FloatMatrix tmp = output_tensor[leave->getGlobalId()].toFloatMatrix(true);
      for (auto &val : tmp) {
        tmp_out(index, 0) = val;
        index++;
      }
    }
    std::cout << tree << std::endl;
    output = tmp_out;
  }

}   // namespace nnet