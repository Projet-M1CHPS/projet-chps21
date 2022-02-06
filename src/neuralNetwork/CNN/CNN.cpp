#include "CNN.hpp"

namespace cnnet {

  void CNN::setTopology(CNNTopology const &topology) {
    std::pair<size_t, size_t> InputSize(topology.getInputSize());

    for (const auto &i : topology) {
      layers.push_back(i->convertToLayer());
      layerMatrix.push_back(FloatMatrix(i->calculateOutputSize(InputSize)));
      // layerMatrix.back().randomize();
      InputSize = std::make_pair(layerMatrix.back().getRows(), layerMatrix.back().getCols());
    }

    this->topology = topology;
  }

  void CNN::setActivationFunction(af::ActivationFunctionType type) { activation_function = type; }

  void CNN::randomizeWeight() {}

  const math::FloatMatrix &CNN::predict(math::FloatMatrix const &input) {
    if (input.getCols() != topology.getInputSize().first or
        input.getRows() != topology.getInputSize().second) {
      throw std::runtime_error("Input size does not match topology input size");
    }

    std::cout << input << std::endl;
    for (auto &i : layerMatrix) std::cout << "\n" << i << std::endl;

    layers[0]->compute(input, layerMatrix[0]);

    for (size_t i = 1; i < layers.size(); i++) {
      layers[i]->compute(layerMatrix[i - 1], layerMatrix[i]);
    }

    std::cout << "\n------------------------------------------------------------------\n" << std::endl;
    std::cout << input << std::endl;
    for (auto &i : layerMatrix) std::cout << "\n" << i << std::endl;

    return layerMatrix.back();
  }

}   // namespace cnnet