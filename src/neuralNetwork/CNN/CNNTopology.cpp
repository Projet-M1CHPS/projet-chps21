#include "CNNTopology.hpp"

namespace cnnet {

  CNNTopologyLayer::CNNTopologyLayer(const std::pair<size_t, size_t> filter, const size_t stride)
      : filter(filter), stride(stride) {}


  CNNTopologyLayerConvolution::CNNTopologyLayerConvolution(const size_t features,
                                                           const std::pair<size_t, size_t> filter,
                                                           const size_t stride,
                                                           const size_t padding)
      : features(features), CNNTopologyLayer(filter, stride), padding(padding) {}

  std::unique_ptr<CNNLayer> CNNTopologyLayerConvolution::convertToLayer() const {
    return std::make_unique<CNNConvolutionLayer>(filter, stride, padding);
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
    os << "Convolution layer: features{" << features << "}, filter{" << filter.first << ", " << filter.second << "}, stride{"
       << stride << "}, padding{" << padding << "}";
    return os;
  }


  CNNTopologyLayerPooling::CNNTopologyLayerPooling(const std::pair<size_t, size_t> filter,
                                                   const size_t stride)
      : CNNTopologyLayer(filter, stride) {}

  std::unique_ptr<CNNLayer> CNNTopologyLayerPooling::convertToLayer() const {
    return std::make_unique<CNNMaxPoolingLayer>(filter, stride);
  }

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

  std::ostream &CNNTopologyLayerPooling::printTo(std::ostream &os) const {
    os << "Pooling layer: filter{" << filter.first << ", " << filter.second << "}, stride{"
       << stride << "}";
    return os;
  }


  CNNTopology::CNNTopology() : inputSize(0, 0) {}
  CNNTopology::CNNTopology(const std::pair<size_t, size_t> &inputSize) : inputSize(inputSize) {}

  const std::shared_ptr<CNNTopologyLayer> &CNNTopology::operator()(size_t index) const
  {
    if(index >= layers.size())
      throw std::out_of_range("Index out of range");
    
    return layers[index];
  }

  void CNNTopology::addConvolution(const size_t features,
                                   const std::pair<size_t, size_t> &filterSize, const size_t stride,
                                   const size_t padding) {
    layers.push_back(
            std::make_shared<CNNTopologyLayerConvolution>(features, filterSize, stride, padding));
  }

  void CNNTopology::addPooling(const std::pair<size_t, size_t> &poolSize, const size_t stride) {
    layers.push_back(std::make_shared<CNNTopologyLayerPooling>(poolSize, stride));
  }


  const CNNTopology stringToTopology(const std::string &str) {
    std::stringstream ss(str);
    size_t inputRow = 0, inputCol = 0;

    ss >> inputRow >> inputCol;
    CNNTopology res(std::make_pair(inputRow, inputCol));

    std::string type;
    while (ss >> type) {
      if (type == "convolution") {
        size_t features = 0, filterRow = 0, filterCol = 0, stride = 0, padding = 0;
        ss >> features >> filterRow >> filterCol >> stride >> padding;
        res.addConvolution(features, {filterRow, filterCol}, stride, padding);
      } else if (type == "pooling") {
        size_t poolRow, poolCol, stride;
        ss >> poolRow >> poolCol >> stride;
        res.addPooling({poolRow, poolCol}, stride);
      } else {
        throw std::invalid_argument("Invalid type " + type);
      }
    }
    return std::move(res);
  }


  std::ostream &operator<<(std::ostream &os, const CNNTopology &nn) {
    os << "input : " << nn.inputSize.first << " " << nn.inputSize.second << "\n";
    for (auto &i : nn.layers) { os << *i << "\n"; }
    return os;
  }
  std::ostream &operator<<(std::ostream &os, const CNNTopologyLayer &layer) {
    return layer.printTo(os);
  }

}   // namespace cnnet