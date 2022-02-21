#include "CNNStorageBP.hpp"


namespace cnnet {

  CNNStorageBP::CNNStorageBP(const std::pair<size_t, size_t> &outputSize) : output(outputSize) {}


  CNNStorageBPConvolution::CNNStorageBPConvolution(const std::pair<size_t, size_t> &outputSize,
                                                   const std::pair<size_t, size_t> &filterSize,
                                                   const size_t stride)
      : CNNStorageBP(outputSize), errorFilter(filterSize),
        dilated4Input(2 * (filterSize.first - 1) + outputSize.first +
                              (stride - 1) * (outputSize.first - 1),
                      2 * (filterSize.second - 1) + outputSize.second +
                              (stride - 1) * (outputSize.second - 1)),
        dilated4Filter(outputSize.first + (stride - 1) * (outputSize.first - 1),
                       outputSize.second + (stride - 1) * (outputSize.second - 1)) {
    for (auto &i : dilated4Input) { i = 0.f; }
    for (auto &i : dilated4Filter) { i = 0.f; }
  }


  CNNStorageBPPooling::CNNStorageBPPooling(const std::pair<size_t, size_t> &outputSize)
      : CNNStorageBP(outputSize) {}

  CNNStorageBPMaxPooling::CNNStorageBPMaxPooling(const std::pair<size_t, size_t> &outputSize)
      : CNNStorageBPPooling(outputSize), maxIndex(outputSize) {}

  CNNStorageBPAvgPooling::CNNStorageBPAvgPooling(const std::pair<size_t, size_t> &outputSize)
      : CNNStorageBPPooling(outputSize) {}

}   // namespace cnnet