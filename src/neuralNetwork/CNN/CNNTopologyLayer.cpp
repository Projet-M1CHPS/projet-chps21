#include "CNNTopologyLayer.hpp"

namespace nnet {

  CNNTopologyLayer::CNNTopologyLayer(const std::pair<size_t, size_t> filter, const size_t stride,
                                     const size_t nBranch)
      : filter(filter), stride(stride), n_branch(nBranch) {}


  const bool CNNTopologyLayer::isValidParameters(const std::pair<size_t, size_t> inputSize,
                                                 const std::pair<size_t, size_t> filterSize,
                                                 const size_t stride, const size_t padding) {
    const size_t resX = (inputSize.first - filterSize.first + (2 * padding)) % stride;
    const size_t resY = (inputSize.second - filterSize.second + (2 * padding)) % stride;
    return resX || resY ? false : true;
  }


  CNNTopologyLayerConvolution::CNNTopologyLayerConvolution(
          const std::pair<size_t, size_t> inputSize, const size_t features,
          const std::pair<size_t, size_t> filter, const af::ActivationFunctionType aFunction,
          const size_t stride, const size_t padding, const size_t nBranch)
      : CNNTopologyLayer(filter, stride, nBranch), features(features), padding(padding),
        activationFunction(aFunction), outputSize(computeOutputSize(inputSize)) {}

  std::shared_ptr<CNNLayer> CNNTopologyLayerConvolution::convertToLayer() const {
    return std::make_shared<CNNConvolutionLayer>(outputSize, filter, features, activationFunction,
                                                 stride, n_branch, padding);
  }

  std::shared_ptr<CNNStorageBP>
  CNNTopologyLayerConvolution::createStorage(const std::pair<size_t, size_t> &inputSize) const {
    return std::make_shared<CNNStorageBPConvolution>();
  }

  const std::pair<size_t, size_t>
  CNNTopologyLayerConvolution::computeOutputSize(const std::pair<size_t, size_t> &inputSize) const {
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
    os << "Convolution layer: nBranch{" << n_branch << "}, outPutSize{" << outputSize.first << ", "
       << outputSize.second << "}, features{" << features << "}, filter{" << filter.first << ", "
       << filter.second << "}, stride{" << stride << "}, padding{" << padding << "}";
    return os;
  }


  CNNTopologyLayerPooling::CNNTopologyLayerPooling(const std::pair<size_t, size_t> inputSize,
                                                   const std::pair<size_t, size_t> filter,
                                                   const size_t stride, const size_t nBranch)
      : CNNTopologyLayer(filter, stride, nBranch), outputSize(computeOutputSize(inputSize)) {}

  const std::pair<size_t, size_t>
  CNNTopologyLayerPooling::computeOutputSize(const std::pair<size_t, size_t> &inputSize) const {
    //(((W - K)/S) + 1)
    // W = Input size
    // K = Filter size
    // S = Stride

    const size_t rows = ((inputSize.first - filter.first) / stride) + 1;
    const size_t cols = ((inputSize.second - filter.second) / stride) + 1;
    return std::make_pair(rows, cols);
  }


  CNNTopologyLayerMaxPooling::CNNTopologyLayerMaxPooling(const std::pair<size_t, size_t> inputSize,
                                                         const std::pair<size_t, size_t> filter,
                                                         const size_t stride, const size_t nBranch)
      : CNNTopologyLayerPooling(inputSize, filter, stride, nBranch) {}

  std::shared_ptr<CNNLayer> CNNTopologyLayerMaxPooling::convertToLayer() const {
    return std::make_shared<CNNMaxPoolingLayer>(outputSize, filter, stride);
  }

  std::shared_ptr<CNNStorageBP>
  CNNTopologyLayerMaxPooling::createStorage(const std::pair<size_t, size_t> &inputSize) const {
    return std::make_shared<CNNStorageBPMaxPooling>(inputSize);
  }

  std::ostream &CNNTopologyLayerMaxPooling::printTo(std::ostream &os) const {
    os << "Max Pooling layer: nBranch{" << n_branch << "}, outputSize{" << outputSize.first << ", "
       << outputSize.second << "}, filter{" << filter.first << ", " << filter.second << "}, stride{"
       << stride << "}";
    return os;
  }


  CNNTopologyLayerAvgPooling::CNNTopologyLayerAvgPooling(const std::pair<size_t, size_t> inputSize,
                                                         const std::pair<size_t, size_t> filter,
                                                         const size_t stride, const size_t nBranch)
      : CNNTopologyLayerPooling(inputSize, filter, stride, nBranch) {}

  std::shared_ptr<CNNLayer> CNNTopologyLayerAvgPooling::convertToLayer() const {
    return std::make_shared<CNNAvgPoolingLayer>(outputSize, filter, stride);
  }

  std::shared_ptr<CNNStorageBP>
  CNNTopologyLayerAvgPooling::createStorage(const std::pair<size_t, size_t> &inputSize) const {
    return std::make_shared<CNNStorageBPAvgPooling>(inputSize);
  }

  std::ostream &CNNTopologyLayerAvgPooling::printTo(std::ostream &os) const {
    os << "Avg Pooling layer: nBranch{" << n_branch << ", outputSize{" << outputSize.first << ", "
       << outputSize.second << "}, nBranch{" << n_branch << "}, filter{" << filter.first << ", "
       << filter.second << "}, stride {" << stride << "}";
    return os;
  }

}   // namespace nnet