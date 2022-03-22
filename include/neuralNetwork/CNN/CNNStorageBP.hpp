#pragma once


#include "Matrix.hpp"
#include <utility>


namespace cnnet {
  using namespace math;

  class CNNStorageBP {
  public:
    CNNStorageBP(const std::pair<size_t, size_t> &inputSize,
                 const std::pair<size_t, size_t> &outputSize);
    ~CNNStorageBP() = default;

    // protected:
    FloatMatrix errorInput;
    FloatMatrix output;
  };

  class CNNStorageBPConvolution final : public CNNStorageBP {
  public:
    CNNStorageBPConvolution(const std::pair<size_t, size_t> &inputSize,
                            const std::pair<size_t, size_t> &outputSize,
                            const std::pair<size_t, size_t> &filterSize, const size_t stride);
    ~CNNStorageBPConvolution() = default;

    // private:
    FloatMatrix errorFilter;
    FloatMatrix dilated4Input;
    FloatMatrix dilated4Filter;
  };

  class CNNStorageBPPooling : public CNNStorageBP {
  public:
    CNNStorageBPPooling(const std::pair<size_t, size_t> &inputSize,
                        const std::pair<size_t, size_t> &outputSize);
    ~CNNStorageBPPooling() = default;
  };

  class CNNStorageBPMaxPooling final : public CNNStorageBPPooling {
  public:
    CNNStorageBPMaxPooling(const std::pair<size_t, size_t> &inputSize,
                           const std::pair<size_t, size_t> &outputSize);
    ~CNNStorageBPMaxPooling() = default;

    // private:
    Matrix<std::pair<size_t, size_t>> maxIndex;
  };

  class CNNStorageBPAvgPooling final : public CNNStorageBPPooling {
  public:
    CNNStorageBPAvgPooling(const std::pair<size_t, size_t> &inputSize,
                           const std::pair<size_t, size_t> &outputSize);
    ~CNNStorageBPAvgPooling() = default;
  };

}   // namespace cnnet