#include "CNN.hpp"

namespace nnet {

  void CNN::setTopology(CNNTopology const &cnn_topology) {
    topology = cnn_topology;
    layers = topology.convertToLayer();
  }


  void CNN::randomizeWeight() { assert(false && "Not implemented"); }


  clFTensor CNN::predict(clFTensor const &input) {
    if (layers.empty()) { throw std::runtime_error("no layer in cnn"); }

    clFTensor output = input.shallowCopy();

    for (auto &layer : layers) {
      output = layer->compute(output);
      for(size_t i = 0; i < output.getZ(); i++)
        std::cout << "output INTERMEDIAIRE " << i << "\n" << output.getMatrix(i).toFloatMatrix(true) << std::endl;
    }

    utils::cl_wrapper.getDefaultQueue().finish();

    return output.flatten();
  }

}   // namespace nnet