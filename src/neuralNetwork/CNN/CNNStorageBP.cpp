#include "CNNStorageBP.hpp"


namespace cnnet {

  CNNStorageBP::CNNStorageBP(const std::pair<size_t, size_t> &inputSize,
                             const std::pair<size_t, size_t> &outputSize)
      : errorInput(inputSize), output(outputSize) {}


  CNNStorageBPConvolution::CNNStorageBPConvolution(const std::pair<size_t, size_t> &inputSize,
                                                   const std::pair<size_t, size_t> &outputSize,
                                                   const std::pair<size_t, size_t> &filterSize)
      : CNNStorageBP(inputSize, outputSize), errorFilter(filterSize) {}


  CNNStorageBPPooling::CNNStorageBPPooling(const std::pair<size_t, size_t> &inputSize,
                                           const std::pair<size_t, size_t> &outputSize)
      : CNNStorageBP(inputSize, outputSize) {}

  CNNStorageBPMaxPooling::CNNStorageBPMaxPooling(const std::pair<size_t, size_t> &inputSize,
                                                 const std::pair<size_t, size_t> &outputSize)
      : CNNStorageBPPooling(inputSize, outputSize), maxIndex(outputSize) {}

  CNNStorageBPAvgPooling::CNNStorageBPAvgPooling(const std::pair<size_t, size_t> &inputSize,
                                                 const std::pair<size_t, size_t> &outputSize)
      : CNNStorageBPPooling(inputSize, outputSize) {}

}   // namespace cnnet