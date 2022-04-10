#pragma once


#include "Matrix.hpp"
#include "clUtils/clFTensor.hpp"
#include <utility>


namespace nnet {
  using namespace math;

  class CNNStorageBP {
  public:
    CNNStorageBP() = default;
    ~CNNStorageBP() = default;
  };

  class CNNStorageBPConvolution final : public CNNStorageBP {
  public:
    CNNStorageBPConvolution() = default;
    ~CNNStorageBPConvolution() = default;

    // private:
    clFTensor errorFilter;
  };

  class CNNStorageBPPooling : public CNNStorageBP {
  public:
    CNNStorageBPPooling(const std::pair<size_t, size_t> inputSize) : input_size(inputSize) {}
    ~CNNStorageBPPooling() = default;

    // private:
    std::pair<size_t, size_t> input_size;
  };

  class CNNStorageBPMaxPooling final : public CNNStorageBPPooling {
  public:
    explicit CNNStorageBPMaxPooling(const std::pair<size_t, size_t> inputSize)
        : CNNStorageBPPooling(inputSize) {}
    ~CNNStorageBPMaxPooling() = default;

    // private:
    // Matrix<std::pair<size_t, size_t>> maxIndex;
  };

  class CNNStorageBPAvgPooling final : public CNNStorageBPPooling {
  public:
    explicit CNNStorageBPAvgPooling(const std::pair<size_t, size_t> inputSize)
        : CNNStorageBPPooling(inputSize) {}
    ~CNNStorageBPAvgPooling() = default;
  };

}   // namespace nnet