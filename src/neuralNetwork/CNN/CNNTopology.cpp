#include "CNNTopology.hpp"

namespace cnnet {

  CNNTopologyLayer::CNNTopologyLayer(const std::pair<size_t, size_t> filter, const size_t stride)
      : filter(filter), stride(stride) {}


  CNNTopologyLayerConvolution::CNNTopologyLayerConvolution(
          const size_t features, const std::pair<size_t, size_t> filter,
          const af::ActivationFunctionType aFunction, const size_t stride, const size_t padding)
      : features(features), CNNTopologyLayer(filter, stride), padding(padding),
        activationFunction(aFunction) {}

  std::shared_ptr<CNNLayer> CNNTopologyLayerConvolution::convertToLayer() const {
    return std::make_shared<CNNConvolutionLayer>(filter, activationFunction, stride, padding);
  }

  std::shared_ptr<CNNStorageBP>
  CNNTopologyLayerConvolution::createStorage(const std::pair<size_t, size_t> &inputSize) const {
    return std::make_shared<CNNStorageBPConvolution>(calculateOutputSize(inputSize), filter,
                                                     stride);
  }

  const std::pair<size_t, size_t> CNNTopologyLayerConvolution::calculateOutputSize(
          const std::pair<size_t, size_t> &inputSize) const {
    //(((W - K + 2P)/S) + 1)
    // W = Input size
    // K = Filter size
    // S = Stride
    // P = Padding

    const size_t rows = ((inputSize.first - filter.first + 2 * padding) / stride) + 1;
    const size_t cols = ((inputSize.second - filter.second + 2 * padding) / stride) + 1;
    return std::make_pair(rows, cols);
  }

  std::ostream &CNNTopologyLayerConvolution::printTo(std::ostream &os) const {
    os << "Convolution layer: features{" << features << "}, filter{" << filter.first << ", "
       << filter.second << "}, stride{" << stride << "}, padding{" << padding << "}";
    return os;
  }


  CNNTopologyLayerPooling::CNNTopologyLayerPooling(const std::pair<size_t, size_t> filter,
                                                   const size_t stride)
      : CNNTopologyLayer(filter, stride) {}

  const std::pair<size_t, size_t>
  CNNTopologyLayerPooling::calculateOutputSize(const std::pair<size_t, size_t> &inputSize) const {
    //(((W - K)/S) + 1)
    // W = Input size
    // K = Filter size
    // S = Stride

    const size_t rows = ((inputSize.first - filter.first) / stride) + 1;
    const size_t cols = ((inputSize.second - filter.second) / stride) + 1;
    return std::make_pair(rows, cols);
  }


  CNNTopologyLayerMaxPooling::CNNTopologyLayerMaxPooling(const std::pair<size_t, size_t> filter,
                                                         const size_t stride)
      : CNNTopologyLayerPooling(filter, stride) {}

  std::shared_ptr<CNNLayer> CNNTopologyLayerMaxPooling::convertToLayer() const {
    return std::make_shared<CNNMaxPoolingLayer>(filter, stride);
  }

  std::shared_ptr<CNNStorageBP>
  CNNTopologyLayerMaxPooling::createStorage(const std::pair<size_t, size_t> &inputSize) const {
    return std::make_shared<CNNStorageBPMaxPooling>(calculateOutputSize(inputSize));
  }

  std::ostream &CNNTopologyLayerMaxPooling::printTo(std::ostream &os) const {
    os << "Max Pooling layer: filter{" << filter.first << ", " << filter.second << "}, stride{"
       << stride << "}";
    return os;
  }


  CNNTopologyLayerAvgPooling::CNNTopologyLayerAvgPooling(const std::pair<size_t, size_t> filter,
                                                         const size_t stride)
      : CNNTopologyLayerPooling(filter, stride) {}

  std::shared_ptr<CNNLayer> CNNTopologyLayerAvgPooling::convertToLayer() const {
    return std::make_shared<CNNAvgPoolingLayer>(filter, stride);
  }

  std::shared_ptr<CNNStorageBP>
  CNNTopologyLayerAvgPooling::createStorage(const std::pair<size_t, size_t> &inputSize) const {
    return std::make_shared<CNNStorageBPAvgPooling>(calculateOutputSize(inputSize));
  }

  std::ostream &CNNTopologyLayerAvgPooling::printTo(std::ostream &os) const {
    os << "Avg Pooling layer: filter{" << filter.first << ", " << filter.second << "}, stride{"
       << stride << "}";
    return os;
  }


  CNNTopology::CNNTopology()
      : inputSize(0, 0), activationFunction(af::ActivationFunctionType::sigmoid) {}
  CNNTopology::CNNTopology(const std::pair<size_t, size_t> &inputSize)
      : inputSize(inputSize), activationFunction(af::ActivationFunctionType::sigmoid) {}

  const std::shared_ptr<CNNTopologyLayer> &CNNTopology::operator()(size_t index) const {
    if (index >= layers.size()) throw std::out_of_range("Index out of range");

    return layers[index];
  }

  void CNNTopology::addConvolution(const size_t features,
                                   const std::pair<size_t, size_t> &filterSize,
                                   const af::ActivationFunctionType aFunction, const size_t stride,
                                   const size_t padding) {
    layers.push_back(std::make_shared<CNNTopologyLayerConvolution>(features, filterSize, aFunction,
                                                                   stride, padding));
  }

  void CNNTopology::addPooling(const PoolingType poolingType,
                               const std::pair<size_t, size_t> &poolSize, const size_t stride) {
    switch (poolingType) {
      case PoolingType::MAX:
        layers.push_back(std::make_shared<CNNTopologyLayerMaxPooling>(poolSize, stride));
        break;
      case PoolingType::AVERAGE:
        layers.push_back(std::make_shared<CNNTopologyLayerAvgPooling>(poolSize, stride));
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
    size_t inputRow = 0, inputCol = 0;

    ss >> inputRow >> inputCol;
    CNNTopology res(std::make_pair(inputRow, inputCol));

    std::string type;
    // read activation function type and store it
    // ss >> type;
    res.activationFunction =
            af::ActivationFunctionType::sigmoid;   // stringToActivationFunctionType(type);

    while (ss >> type) {
      LayerType layerType = stringToLayerType(type);
      if (layerType == LayerType::CONVOLUTION) {
        size_t features = 0, filterRow = 0, filterCol = 0, stride = 0, padding = 0;
        ss >> features >> filterRow >> filterCol >> stride >> padding;
        res.addConvolution(features, {filterRow, filterCol}, res.activationFunction, stride,
                           padding);
      } else if (layerType == LayerType::POOLING) {
        std::string strPoolingType;
        size_t poolRow, poolCol, stride;
        ss >> strPoolingType >> poolRow >> poolCol >> stride;
        PoolingType poolType = stringToPoolingType(strPoolingType);
        res.addPooling(poolType, {poolRow, poolCol}, stride);
      } else {
        throw std::invalid_argument("Invalid type " + type);
      }
    }
    return std::move(res);
  }


  std::ostream &operator<<(std::ostream &os, const CNNTopology &nn) {
    os << "input : " << nn.inputSize.first << " " << nn.inputSize.second << "\n";
    os << "activation function : " << /*af::AFTypeToStr(nn.activationFunction) <<*/ "\n";
    for (auto &i : nn.layers) { os << *i << "\n"; }
    return os;
  }
  std::ostream &operator<<(std::ostream &os, const CNNTopologyLayer &layer) {
    return layer.printTo(os);
  }

}   // namespace cnnet