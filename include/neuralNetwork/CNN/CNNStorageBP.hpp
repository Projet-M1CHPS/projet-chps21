#pragma once


#include "Matrix.hpp"
#include "math/clFTensor.hpp"
#include <utility>


namespace nnet {

  class CNNStorageBP {
  public:
    CNNStorageBP() = default;
    virtual ~CNNStorageBP() = default;

    /**
     * @brief Check if storage have weight
     * @return true if layer have weight, otherwise false
     */
    virtual bool hasGradient() const { return false; }

    /**
     * @brief Getter to retrieve weight
     * @return return weight if they exist otherwise throw an assertion
     */
    virtual math::clFTensor &getGradient() {
      throw std::runtime_error("CNNStorageBP: Tried to acces gradient in a storage without one");
    }

    /**
     * @brief Getter to retrieve weight
     * @return return weight if they exist otherwise throw an assertion
     */
    virtual const math::clFTensor &getGradient() const {
      throw std::runtime_error("CNNStorageBP: Tried to acces gradient in a storage without one");
    }
  };

  class CNNStorageBPConvolution final : public CNNStorageBP {
  public:
    CNNStorageBPConvolution() = default;

    /**
     * @brief Check if storage have weight
     * @return true if layer have weight, otherwise false
     */
    bool hasGradient() const override { return true; }


    math::clFTensor &getGradient() override { return error_filter; }

    /**
     * @brief Getter to retrieve weight
     * @return return weight if they exist otherwise throw an assertion
     */
    const math::clFTensor &getGradient() const override { return error_filter; }

    math::clFTensor input;
    math::clFTensor error_filter;
  };

  class CNNStorageBPPooling : public CNNStorageBP {
  public:
    explicit CNNStorageBPPooling(const std::pair<size_t, size_t> inputSize)
        : input_size(inputSize) {}

    std::pair<size_t, size_t> input_size;
  };

  class CNNStorageBPMaxPooling final : public CNNStorageBPPooling {
  public:
    explicit CNNStorageBPMaxPooling(const std::pair<size_t, size_t> inputSize)
        : CNNStorageBPPooling(inputSize) {}

    std::vector<math::Matrix<size_t>> max_rows;
    std::vector<math::Matrix<size_t>> max_cols;
  };

  class CNNStorageBPAvgPooling final : public CNNStorageBPPooling {
  public:
    explicit CNNStorageBPAvgPooling(const std::pair<size_t, size_t> inputSize)
        : CNNStorageBPPooling(inputSize) {}
  };


}   // namespace nnet