#pragma once


#include "Matrix.hpp"
#include "math/clFTensor.hpp"
#include <utility>


namespace nnet {
  using namespace math;

  class CNNStorageBP {
  public:
    CNNStorageBP() = default;
    virtual ~CNNStorageBP() = default;

    // Pretty messy but easiest way to do it for now
    virtual bool hasGradient() const { return false; }

    virtual clFTensor &getGradient() {
      throw std::runtime_error("CNNStorageBP: Tried to acces gradient in a storage without one");
    }

    virtual const clFTensor &getGradient() const {
      throw std::runtime_error("CNNStorageBP: Tried to acces gradient in a storage without one");
    }
  };

  class CNNStorageBPConvolution final : public CNNStorageBP {
  public:
    CNNStorageBPConvolution() = default;

    bool hasGradient() const override { return true; }

    clFTensor &getGradient() override { return error_filter; }
    const clFTensor &getGradient() const override { return error_filter; }

    // private:
    clFTensor input;
    clFTensor error_filter;
  };

  class CNNStorageBPPooling : public CNNStorageBP {
  public:
    explicit CNNStorageBPPooling(const std::pair<size_t, size_t> inputSize)
        : input_size(inputSize) {}

    // private:
    std::pair<size_t, size_t> input_size;
  };

  class CNNStorageBPMaxPooling final : public CNNStorageBPPooling {
  public:
    explicit CNNStorageBPMaxPooling(const std::pair<size_t, size_t> inputSize)
        : CNNStorageBPPooling(inputSize) {}

    // private:
    // Matrix<std::pair<size_t, size_t>> maxIndex;
    std::vector<Matrix<size_t>> max_rows;
    std::vector<Matrix<size_t>> max_cols;
  };

  class CNNStorageBPAvgPooling final : public CNNStorageBPPooling {
  public:
    explicit CNNStorageBPAvgPooling(const std::pair<size_t, size_t> inputSize)
        : CNNStorageBPPooling(inputSize) {}
  };


}   // namespace nnet