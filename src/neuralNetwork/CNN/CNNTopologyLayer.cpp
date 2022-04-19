#include "CNNTopologyLayer.hpp"

namespace nnet {

  CNNTopologyLayer::CNNTopologyLayer(const std::pair<size_t, size_t> inputSize, const std::pair<size_t, size_t> filter, const size_t nBranch)
      : input_size(inputSize), filter(filter), n_branch(nBranch) {}


  CNNTopologyLayerConvolution::CNNTopologyLayerConvolution(
          const std::pair<size_t, size_t> inputSize, const size_t features,
          const std::pair<size_t, size_t> filter, const af::ActivationFunctionType aFunction,
          const size_t nBranch)
      : CNNTopologyLayer(inputSize, filter, nBranch), features(features), activationFunction(aFunction),
        outputSize(computeOutputSize(inputSize)) {}

  std::unique_ptr<CNNLayer> CNNTopologyLayerConvolution::convertToLayer() const {
    return std::make_unique<CNNConvolutionLayer>(outputSize, filter, features, activationFunction,
                                                 n_branch);
  }

  std::unique_ptr<CNNStorageBP> CNNTopologyLayerConvolution::convertToStorage() const {
    return std::make_unique<CNNStorageBPConvolution>();
  }

  const std::pair<size_t, size_t>
  CNNTopologyLayerConvolution::computeOutputSize(const std::pair<size_t, size_t> &inputSize) const {
    //(((W - K + 2P)/S) + 1)
    // W = Input size
    // K = Filter size
    // S = Stride ( 1 )
    // P = Padding ( 0 )
    const size_t rows = (inputSize.first - filter.first) + 1;
    const size_t cols = (inputSize.second - filter.second) + 1;
    return std::make_pair(rows, cols);
  }

  std::ostream &CNNTopologyLayerConvolution::printTo(std::ostream &os) const {
    os << "Convolution layer: nBranch{" << n_branch << "}, outPutSize{" << outputSize.first << ", "
       << outputSize.second << "}, features{" << features << "}, filter{" << filter.first << ", "
       << filter.second << "}";
    return os;
  }


  CNNTopologyLayerPooling::CNNTopologyLayerPooling(const std::pair<size_t, size_t> inputSize,
                                                   const std::pair<size_t, size_t> filter,
                                                   const size_t nBranch)
      : CNNTopologyLayer(inputSize, filter, nBranch), outputSize(computeOutputSize(inputSize)) {}

  const std::pair<size_t, size_t>
  CNNTopologyLayerPooling::computeOutputSize(const std::pair<size_t, size_t> &inputSize) const {
    //(((W - K)/S) + 1)
    // W = Input size
    // K = Filter size
    // S = Stride ( 1 )

    const size_t rows = (inputSize.first - filter.first) + 1;
    const size_t cols = (inputSize.second - filter.second) + 1;
    return std::make_pair(rows, cols);
  }


  CNNTopologyLayerMaxPooling::CNNTopologyLayerMaxPooling(const std::pair<size_t, size_t> inputSize,
                                                         const std::pair<size_t, size_t> filter,
                                                         const size_t nBranch)
      : CNNTopologyLayerPooling(inputSize, filter, nBranch) {}

  std::unique_ptr<CNNLayer> CNNTopologyLayerMaxPooling::convertToLayer() const {
    return std::make_unique<CNNMaxPoolingLayer>(outputSize, filter);
  }

  std::unique_ptr<CNNStorageBP> CNNTopologyLayerMaxPooling::convertToStorage() const {
    return std::make_unique<CNNStorageBPMaxPooling>(input_size);
  }

  std::ostream &CNNTopologyLayerMaxPooling::printTo(std::ostream &os) const {
    os << "Max Pooling layer: nBranch{" << n_branch << "}, outputSize{" << outputSize.first << ", "
       << outputSize.second << "}, filter{" << filter.first << ", " << filter.second << "}";
    return os;
  }


  CNNTopologyLayerAvgPooling::CNNTopologyLayerAvgPooling(const std::pair<size_t, size_t> inputSize,
                                                         const std::pair<size_t, size_t> filter,
                                                         const size_t nBranch)
      : CNNTopologyLayerPooling(inputSize, filter, nBranch) {}

  std::unique_ptr<CNNLayer> CNNTopologyLayerAvgPooling::convertToLayer() const {
    return std::make_unique<CNNAvgPoolingLayer>(outputSize, filter);
  }

  std::unique_ptr<CNNStorageBP> CNNTopologyLayerAvgPooling::convertToStorage() const {
    return std::make_unique<CNNStorageBPAvgPooling>(input_size);
  }

  std::ostream &CNNTopologyLayerAvgPooling::printTo(std::ostream &os) const {
    os << "Avg Pooling layer: nBranch{" << n_branch << ", outputSize{" << outputSize.first << ", "
       << outputSize.second << "}, nBranch{" << n_branch << "}, filter{" << filter.first << ", "
       << filter.second << "}";
    return os;
  }

}   // namespace nnet