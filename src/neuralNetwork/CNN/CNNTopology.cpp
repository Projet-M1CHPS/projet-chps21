#include "CNNTopology.hpp"

namespace nnet {

  CNNTopology::CNNTopology()
      : inputSize(0, 0), activationFunction(af::ActivationFunctionType::sigmoid) {}
  CNNTopology::CNNTopology(const std::pair<size_t, size_t> &inputSize)
      : inputSize(inputSize), activationFunction(af::ActivationFunctionType::sigmoid) {}

  const std::shared_ptr<CNNTopologyLayer> &CNNTopology::operator()(size_t index) const {
    if (index >= layers.size()) throw std::out_of_range("Index out of range");

    return layers[index];
  }

  std::vector<std::shared_ptr<CNNLayer>> CNNTopology::convertToLayer() {
    std::vector<std::shared_ptr<CNNLayer>> res;
    for (auto &layer : layers) { res.push_back(layer->convertToLayer()); }
    return res;
  }

  void CNNTopology::addConvolution(const std::pair<size_t, size_t> &inputSize,
                                   const size_t features,
                                   const std::pair<size_t, size_t> &filterSize,
                                   const af::ActivationFunctionType aFunction, const size_t stride,
                                   const size_t padding, const size_t nbranch) {
    layers.push_back(std::make_shared<CNNTopologyLayerConvolution>(
            inputSize, features, filterSize, aFunction, stride, padding, nbranch));
  }

  void CNNTopology::addPooling(const std::pair<size_t, size_t> &inputSize,
                               const PoolingType poolingType,
                               const std::pair<size_t, size_t> &poolSize, const size_t stride,
                               const size_t nbranch) {
    switch (poolingType) {
      case PoolingType::MAX:
        layers.push_back(
                std::make_shared<CNNTopologyLayerMaxPooling>(inputSize, poolSize, stride, nbranch));
        break;
      case PoolingType::AVERAGE:
        layers.push_back(
                std::make_shared<CNNTopologyLayerAvgPooling>(inputSize, poolSize, stride, nbranch));
        break;
      default:
        throw std::invalid_argument("Invalid pooling type");
    }
  }


  const LayerType stringToLayerType(const std::string &str) {
    if (str == "convolution") return LayerType::CONVOLUTION;
    else if (str == "pooling")
      return LayerType::POOLING;
    else
      throw std::invalid_argument("Unknown layer type");
  }

  const PoolingType stringToPoolingType(const std::string &str) {
    if (str == "max") return PoolingType::MAX;
    else if (str == "avg")
      return PoolingType::AVERAGE;
    else
      throw std::invalid_argument("Unknown pooling type");
  }

  const CNNTopology stringToTopology(const std::string &str) {
    std::stringstream ss(str);
    std::pair<size_t, size_t> inputSize = {0, 0};

    ss >> inputSize.first >> inputSize.second;
    CNNTopology res(inputSize);

    std::string type;
    ss >> type;
    res.activationFunction = af::strToAFType(type);

    size_t n_branch = 1;
    while (ss >> type) {
      LayerType layerType = stringToLayerType(type);
      if (layerType == LayerType::CONVOLUTION) {
        size_t features = 0, stride = 0, padding = 0;
        std::pair<size_t, size_t> filterSize = {0, 0};
        ss >> features >> filterSize.first >> filterSize.second >> stride >> padding;
        if (not CNNTopologyLayer::isValidParameters(inputSize, filterSize, stride, padding))
          throw std::runtime_error("Invalid parameters for convolution layer");
        res.addConvolution(inputSize, features, filterSize, res.activationFunction, stride,
                           padding, n_branch);
        n_branch *= features;
      } else if (layerType == LayerType::POOLING) {
        std::string strPoolingType;
        std::pair<size_t, size_t> poolSize;
        size_t stride;
        ss >> strPoolingType >> poolSize.first >> poolSize.second >> stride;
        if (not CNNTopologyLayer::isValidParameters(inputSize, poolSize, stride, 0))
          throw std::runtime_error("Invalid parameters for pooling layer");
        PoolingType poolType = stringToPoolingType(strPoolingType);
        res.addPooling(inputSize, poolType, poolSize, stride, n_branch);
      } else {
        throw std::invalid_argument("Invalid type " + type);
      }
      inputSize = res.layers.back()->getOutputSize(inputSize);
    }
    return std::move(res);
  }


  std::ostream &operator<<(std::ostream &os, const CNNTopology &nn) {
    os << "input : " << nn.inputSize.first << " " << nn.inputSize.second << "\n";
    os << "activation function : " << af::AFTypeToStr(nn.activationFunction) << "\n";
    for (auto &i : nn.layers) { os << *i << "\n"; }
    return os;
  }
  std::ostream &operator<<(std::ostream &os, const CNNTopologyLayer &layer) {
    return layer.printTo(os);
  }

}   // namespace nnet