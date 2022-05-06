#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "ActivationFunction.hpp"
#include "CNNLayer.hpp"
#include "CNNStorageBP.hpp"

namespace nnet {

  /**
   * @brief Base class to describe layer into the topology of cnn
   */
  class CNNTopologyLayer {
    friend std::ostream &operator<<(std::ostream &os, const CNNTopologyLayer &layer);

  public:
    CNNTopologyLayer(const std::pair<size_t, size_t> inputSize,
                     const std::pair<size_t, size_t> filter, const size_t nbranch);
    ~CNNTopologyLayer() = default;

    [[nodiscard]] const std::pair<size_t, size_t> &getFilterSize() const { return filter_size; }
    [[nodiscard]] virtual const size_t getFeatures() const { return 1; }

    /**
     * @brief Convert the topology layer into CNNLayer
     * @return Converted layer topology into layer
     */
    [[nodiscard]] virtual std::unique_ptr<CNNLayer> convertToLayer() const = 0;

    /**
     * @brief Convert the topology layer into CNNStorageBP
     * @return Converted layer topology into storage
     */
    [[nodiscard]] virtual std::unique_ptr<CNNStorageBP> convertToStorage() const = 0;

    [[nodiscard]] virtual const std::pair<size_t, size_t> getOutputSize() const = 0;

  protected:
    /**
     * @brief Compute the output size
     * @param inputSize Image size
     * @return Size of output image with transformation
     */
    [[nodiscard]] virtual const std::pair<size_t, size_t>
    computeOutputSize(const std::pair<size_t, size_t> &inputSize) const = 0;
    virtual std::ostream &printTo(std::ostream &) const = 0;

  protected:
    const std::pair<size_t, size_t> input_size;
    const std::pair<size_t, size_t> filter_size;
    const size_t n_branch;
  };

  /**
   * @brief Class to describe convolutional layer into the topology of cnn
   */
  class CNNTopologyLayerConvolution final : public CNNTopologyLayer {
  public:
    CNNTopologyLayerConvolution(const std::pair<size_t, size_t> inputSize, const size_t features,
                                const std::pair<size_t, size_t> filter,
                                const af::ActivationFunctionType aFunction, const size_t nBranch);
    ~CNNTopologyLayerConvolution() = default;

    [[nodiscard]] const size_t getFeatures() const override { return features; }

    /**
     * @brief Convert the topology layer into CNNLayer
     * @return Converted layer topology into layer
     */
    [[nodiscard]] std::unique_ptr<CNNLayer> convertToLayer() const override;

    /**
     * @brief Convert the topology layer into CNNStorageBP
     * @return Converted layer topology into storage
     */
    [[nodiscard]] std::unique_ptr<CNNStorageBP> convertToStorage() const override;

    [[nodiscard]] const std::pair<size_t, size_t> getOutputSize() const override {
      return outputSize;
    };

  private:
    /**
     * @brief Compute the output size
     * @param inputSize Image size
     * @return Size of output image with transformation
     */
    [[nodiscard]] const std::pair<size_t, size_t>
    computeOutputSize(const std::pair<size_t, size_t> &inputSize) const override;
    std::ostream &printTo(std::ostream &) const override;

  private:
    const size_t features;
    const af::ActivationFunctionType activationFunction;
    const std::pair<size_t, size_t> outputSize;
  };

  /**
   * @brief Base class to describe pooling layer into the topology of cnn
   */
  class CNNTopologyLayerPooling : public CNNTopologyLayer {
  public:
    CNNTopologyLayerPooling(const std::pair<size_t, size_t> inputSize,
                            const std::pair<size_t, size_t> filter, const size_t nBranch);
    ~CNNTopologyLayerPooling() = default;


    [[nodiscard]] const std::pair<size_t, size_t> getOutputSize() const override {
      return outputSize;
    };

  private:
    /**
     * @brief Compute the output size
     * @param inputSize Image size
     * @return Size of output image with transformation
     */
    [[nodiscard]] const std::pair<size_t, size_t>
    computeOutputSize(const std::pair<size_t, size_t> &inputSize) const override;
    virtual std::ostream &printTo(std::ostream &) const = 0;

  protected:
    const std::pair<size_t, size_t> outputSize;
  };

  /**
   * @brief Base class to describe max pooling layer into the topology of cnn
   */
  class CNNTopologyLayerMaxPooling final : public CNNTopologyLayerPooling {
  public:
    CNNTopologyLayerMaxPooling(const std::pair<size_t, size_t> inputSize,
                               const std::pair<size_t, size_t> filter, const size_t nBranch);
    ~CNNTopologyLayerMaxPooling() = default;

    /**
     * @brief Convert the topology layer into CNNLayer
     * @return Converted layer topology into layer
     */
    [[nodiscard]] std::unique_ptr<CNNLayer> convertToLayer() const override;

    /**
     * @brief Convert the topology layer into CNNStorageBP
     * @return Converted layer topology into storage
     */
    [[nodiscard]] std::unique_ptr<CNNStorageBP> convertToStorage() const override;

  private:
    std::ostream &printTo(std::ostream &) const override;
  };


  /**
   * @brief Base class to describe average pooling layer into the topology of cnn
   */
  class CNNTopologyLayerAvgPooling final : public CNNTopologyLayerPooling {
  public:
    CNNTopologyLayerAvgPooling(const std::pair<size_t, size_t> inputSize,
                               const std::pair<size_t, size_t> filter, const size_t nBranch);
    ~CNNTopologyLayerAvgPooling() = default;

    /**
     * @brief Convert the topology layer into CNNLayer
     * @return Converted layer topology into layer
     */
    [[nodiscard]] std::unique_ptr<CNNLayer> convertToLayer() const override;

    /**
     * @brief Convert the topology layer into CNNStorageBP
     * @return Converted layer topology into storage
     */
    [[nodiscard]] std::unique_ptr<CNNStorageBP> convertToStorage() const override;

  private:
    std::ostream &printTo(std::ostream &) const override;
  };

}   // namespace nnet