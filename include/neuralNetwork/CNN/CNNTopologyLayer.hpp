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

  class CNNTopologyLayer {
    friend std::ostream &operator<<(std::ostream &os, const CNNTopologyLayer &layer);

  public:
    CNNTopologyLayer(const std::pair<size_t, size_t> inputSize, const std::pair<size_t, size_t> filter, const size_t nbranch);
    ~CNNTopologyLayer() = default;

    [[nodiscard]] const std::pair<size_t, size_t> &getFilterSize() const { return filter; }
    [[nodiscard]] virtual const size_t getFeatures() const { return 1; }

    [[nodiscard]] virtual std::unique_ptr<CNNLayer> convertToLayer() const = 0;
    [[nodiscard]] virtual std::unique_ptr<CNNStorageBP> convertToStorage() const = 0;

    [[nodiscard]] virtual const std::pair<size_t, size_t>
    getOutputSize(const std::pair<size_t, size_t> &inputSize) const = 0;

  protected:
    [[nodiscard]] virtual const std::pair<size_t, size_t>
    computeOutputSize(const std::pair<size_t, size_t> &inputSize) const = 0;
    virtual std::ostream &printTo(std::ostream &) const = 0;

  protected:
    const std::pair<size_t, size_t> input_size;
    const std::pair<size_t, size_t> filter;
    const size_t n_branch;
  };


  class CNNTopologyLayerConvolution final : public CNNTopologyLayer {
  public:
    CNNTopologyLayerConvolution(const std::pair<size_t, size_t> inputSize, const size_t features,
                                const std::pair<size_t, size_t> filter,
                                const af::ActivationFunctionType aFunction, const size_t nBranch);
    ~CNNTopologyLayerConvolution() = default;

    [[nodiscard]] const size_t getFeatures() const override { return features; }

    [[nodiscard]] std::unique_ptr<CNNLayer> convertToLayer() const override;
    [[nodiscard]] std::unique_ptr<CNNStorageBP> convertToStorage() const override;

    [[nodiscard]] const std::pair<size_t, size_t>
    getOutputSize(const std::pair<size_t, size_t> &inputSize) const override {
      return outputSize;
    };

  private:
    [[nodiscard]] const std::pair<size_t, size_t>
    computeOutputSize(const std::pair<size_t, size_t> &inputSize) const override;
    std::ostream &printTo(std::ostream &) const override;

  private:
    const size_t features;
    const af::ActivationFunctionType activationFunction;
    const std::pair<size_t, size_t> outputSize;
  };


  class CNNTopologyLayerPooling : public CNNTopologyLayer {
  public:
    CNNTopologyLayerPooling(const std::pair<size_t, size_t> inputSize,
                            const std::pair<size_t, size_t> filter, const size_t nBranch);
    ~CNNTopologyLayerPooling() = default;


    [[nodiscard]] const std::pair<size_t, size_t>
    getOutputSize(const std::pair<size_t, size_t> &inputSize) const override {
      return outputSize;
    };

  private:
    [[nodiscard]] const std::pair<size_t, size_t>
    computeOutputSize(const std::pair<size_t, size_t> &inputSize) const override;
    virtual std::ostream &printTo(std::ostream &) const = 0;

  protected:
    const std::pair<size_t, size_t> outputSize;
  };


  class CNNTopologyLayerMaxPooling final : public CNNTopologyLayerPooling {
  public:
    CNNTopologyLayerMaxPooling(const std::pair<size_t, size_t> inputSize,
                               const std::pair<size_t, size_t> filter, const size_t nBranch);
    ~CNNTopologyLayerMaxPooling() = default;

    [[nodiscard]] std::unique_ptr<CNNLayer> convertToLayer() const override;
    [[nodiscard]] std::unique_ptr<CNNStorageBP> convertToStorage() const override;

  private:
    std::ostream &printTo(std::ostream &) const override;
  };

  class CNNTopologyLayerAvgPooling final : public CNNTopologyLayerPooling {
  public:
    CNNTopologyLayerAvgPooling(const std::pair<size_t, size_t> inputSize,
                               const std::pair<size_t, size_t> filter, const size_t nBranch);
    ~CNNTopologyLayerAvgPooling() = default;

    [[nodiscard]] std::unique_ptr<CNNLayer> convertToLayer() const override;
    [[nodiscard]] std::unique_ptr<CNNStorageBP> convertToStorage() const override;

  private:
    std::ostream &printTo(std::ostream &) const override;
  };

}   // namespace nnet