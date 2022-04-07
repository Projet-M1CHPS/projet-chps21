#include "CNNStorageBP.hpp"


namespace nnet {

  CNNStorageBP::CNNStorageBP(const std::pair<size_t, size_t> &inputSize,
                             const std::pair<size_t, size_t> &outputSize)
      : errorInput(inputSize.first, inputSize.first, 0),
        output(outputSize.first, outputSize.second, 0) {}


  CNNStorageBPConvolution::CNNStorageBPConvolution(const std::pair<size_t, size_t> &inputSize,
                                                   const std::pair<size_t, size_t> &outputSize,
                                                   const std::pair<size_t, size_t> &filterSize,
                                                   const size_t stride)
      : CNNStorageBP(inputSize, outputSize), errorFilter(filterSize.first, filterSize.second, 0) {}


  CNNStorageBPPooling::CNNStorageBPPooling(const std::pair<size_t, size_t> &inputSize,
                                           const std::pair<size_t, size_t> &outputSize)
      : CNNStorageBP(inputSize, outputSize) {}

  CNNStorageBPMaxPooling::CNNStorageBPMaxPooling(const std::pair<size_t, size_t> &inputSize,
                                                 const std::pair<size_t, size_t> &outputSize)
      : CNNStorageBPPooling(inputSize, outputSize) {}

  CNNStorageBPAvgPooling::CNNStorageBPAvgPooling(const std::pair<size_t, size_t> &inputSize,
                                                 const std::pair<size_t, size_t> &outputSize)
      : CNNStorageBPPooling(inputSize, outputSize) {}

}   // namespace nnet