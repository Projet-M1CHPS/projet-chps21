#include "CNN.hpp"

namespace cnnet {

  CNN::CNN(CNN &&other) noexcept {
    this->topology = std::move(other.topology);
    this->layers = std::move(other.layers);
    this->layerMatrix = std::move(other.layerMatrix);
  }

  CNN &CNN::operator=(CNN &&other) noexcept {
    this->topology = std::move(other.topology);
    this->layers = std::move(other.layers);
    this->layerMatrix = std::move(other.layerMatrix);
    return *this;
  }

  void CNN::setTopology(CNNTopology const &topology) {
    const size_t deepth = topology.getDeepth();

    layers.resize(deepth);
    layerMatrix.resize(deepth);

    layers[0].resize(topology(0)->getFeatures());
    layerMatrix[0].resize(topology(0)->getFeatures());

    for (size_t i = 1; i < deepth; i++) {
      layers[i].resize(topology(i)->getFeatures() * layers[i - 1].size());
      layerMatrix[i].resize(topology(i)->getFeatures() * layerMatrix[i - 1].size());
    }

    std::pair<size_t, size_t> inputSize(topology.getInputSize());
    for (size_t i = 0; i < topology(0)->getFeatures(); i++) {
      layers[0][i] = topology(0)->convertToLayer();
      layerMatrix[0][i] = FloatMatrix(topology(0)->calculateOutputSize(inputSize));
    }

    inputSize = std::make_pair(layerMatrix[0].front().getRows(), layerMatrix[0].front().getCols());

    for (size_t i = 1; i < deepth; i++) {
      inputSize = std::make_pair(layerMatrix[i - 1].front().getRows(),
                                 layerMatrix[i - 1].front().getCols());
      for (size_t j = 0; j < layers[i].size(); j++) {
        layers[i][j] = topology(i)->convertToLayer();
        layerMatrix[i][j] = FloatMatrix(topology(i)->calculateOutputSize(inputSize));
      }
    }

    this->topology = topology;
  }


  void CNN::randomizeWeight() {}

  void CNN::predict(math::FloatMatrix const &input, math::FloatMatrix &output) {
    if (input.getCols() != topology.getInputSize().first or
        input.getRows() != topology.getInputSize().second) {
      throw std::runtime_error("Input size does not match topology input size");
    } else if (output.getCols() != 1 or output.getRows() != getOutputSize()) {
      throw std::runtime_error("Output size does not match topology output size");
    }

    std::cout << "forward \n" << std::endl;
    std::cout << input << std::endl;

    for (size_t i = 0; i < layers.size(); i++) {
      std::cout << "------------------------------------------" << std::endl;
      for (size_t j = 0; j < layers[i].size(); j++)
        std::cout << layerMatrix[i][j] << "\n" << std::endl;
    }

    for (size_t i = 0; i < topology(0)->getFeatures(); i++)
      layers[0][i]->compute(input, layerMatrix[0][i]);

    for (size_t i = 1; i < topology.getDeepth(); i++) {
      size_t l = 0;
      for (size_t j = 0; j < layers[i - 1].size(); j++) {
        for (size_t k = 0; k < topology(i)->getFeatures(); k++) {
          //std::cout << "(" << i - 1 << ", " << j << ") (" << i << ", " << l << ")" << std::endl;
          layers[i][j]->compute(layerMatrix[i - 1][j], layerMatrix[i][l++]);
        }
      }
    }

    size_t index = 0;
    for (auto mat : layerMatrix.back()) {
      for (auto val : mat) {
        output(index, 0) = val;
        index++;
      }
    }

    std::cout << "////////////////////////////////////////////" << std::endl;
    for (size_t i = 0; i < layers.size(); i++) {
      std::cout << "------------------------------------------" << std::endl;
      for (size_t j = 0; j < layers[i].size(); j++)
        std::cout << layerMatrix[i][j] << "\n" << std::endl;
    }
  }

}   // namespace cnnet