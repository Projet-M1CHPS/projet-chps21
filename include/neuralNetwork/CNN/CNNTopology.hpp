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
    CNNTopologyLayer(const std::pair<size_t, size_t> filter, const size_t stride);
    ~CNNTopologyLayer() = default;

    const std::pair<size_t, size_t> &getFilterSize() const { return filter; }
    const size_t &getStride() const { return stride; }
    virtual const size_t getFeatures() const { return 1; }

    virtual std::shared_ptr<CNNLayer> convertToLayer() const = 0;
    virtual std::shared_ptr<CNNStorageBP>
    createStorage(const std::pair<size_t, size_t> &inputSize) const = 0;

    virtual const std::pair<size_t, size_t>
    getOutputSize(const std::pair<size_t, size_t> &inputSize) const {
      return std::make_pair(0, 0);
    };

    static const bool isValidParameters(const std::pair<size_t, size_t> inputSize,
                                        const std::pair<size_t, size_t> filterSize,
                                        const size_t stride, const size_t padding);

  protected:
    virtual const std::pair<size_t, size_t>
    computeOutputSize(const std::pair<size_t, size_t> &inputSize) const = 0;
    virtual std::ostream &printTo(std::ostream &) const = 0;

  protected:
    const std::pair<size_t, size_t> filter;
    const size_t stride;
  };


  class CNNTopologyLayerConvolution final : public CNNTopologyLayer {
  public:
    CNNTopologyLayerConvolution(const std::pair<size_t, size_t> inputSize, const size_t features,
                                const std::pair<size_t, size_t> filter,
                                const af::ActivationFunctionType aFunction, const size_t stride,
                                const size_t padding);
    ~CNNTopologyLayerConvolution() = default;

    const size_t getFeatures() const override { return features; }
    const size_t &getPadding() const { return padding; }

    std::shared_ptr<CNNLayer> convertToLayer() const override;
    std::shared_ptr<CNNStorageBP>
    createStorage(const std::pair<size_t, size_t> &inputSize) const override;

    const std::pair<size_t, size_t>
    getOutputSize(const std::pair<size_t, size_t> &inputSize) const override {
      return outputSize;
    };

  private:
    const std::pair<size_t, size_t>
    computeOutputSize(const std::pair<size_t, size_t> &inputSize) const override;
    std::ostream &printTo(std::ostream &) const override;

  private:
    const size_t features;
    const size_t padding;
    const af::ActivationFunctionType activationFunction;
    const std::pair<size_t, size_t> outputSize;
  };


  class CNNTopologyLayerPooling : public CNNTopologyLayer {
  public:
    CNNTopologyLayerPooling(const std::pair<size_t, size_t> inputSize,
                            const std::pair<size_t, size_t> filter, const size_t stride);
    ~CNNTopologyLayerPooling() = default;

    virtual std::shared_ptr<CNNLayer> convertToLayer() const = 0;
    virtual std::shared_ptr<CNNStorageBP>
    createStorage(const std::pair<size_t, size_t> &inputSize) const = 0;

    const std::pair<size_t, size_t>
    getOutputSize(const std::pair<size_t, size_t> &inputSize) const override {
      return outputSize;
    };

  private:
    const std::pair<size_t, size_t>
    computeOutputSize(const std::pair<size_t, size_t> &inputSize) const override;
    virtual std::ostream &printTo(std::ostream &) const = 0;

  protected:
    const std::pair<size_t, size_t> outputSize;
  };

  class CNNTopologyLayerMaxPooling final : public CNNTopologyLayerPooling {
  public:
    CNNTopologyLayerMaxPooling(const std::pair<size_t, size_t> inputSize,
                               const std::pair<size_t, size_t> filter, const size_t stride);
    ~CNNTopologyLayerMaxPooling() = default;

    std::shared_ptr<CNNLayer> convertToLayer() const override;
    std::shared_ptr<CNNStorageBP>
    createStorage(const std::pair<size_t, size_t> &inputSize) const override;

  private:
    std::ostream &printTo(std::ostream &) const override;
  };

  class CNNTopologyLayerAvgPooling final : public CNNTopologyLayerPooling {
  public:
    CNNTopologyLayerAvgPooling(const std::pair<size_t, size_t> inputSize,
                               const std::pair<size_t, size_t> filter, const size_t stride);
    ~CNNTopologyLayerAvgPooling() = default;

    std::shared_ptr<CNNLayer> convertToLayer() const override;
    std::shared_ptr<CNNStorageBP>
    createStorage(const std::pair<size_t, size_t> &inputSize) const override;

  private:
    std::ostream &printTo(std::ostream &) const override;
  };


  class CNNTopology {
    typedef typename std::vector<std::shared_ptr<CNNTopologyLayer>>::const_iterator const_iterator;
    friend std::ostream &operator<<(std::ostream &os, const CNNTopology &topology);
    friend const CNNTopology stringToTopology(const std::string &str);

  public:
    CNNTopology();
    explicit CNNTopology(const std::pair<size_t, size_t> &inputSize);

    CNNTopology(const CNNTopology &) = default;
    CNNTopology(CNNTopology &&) = default;

    CNNTopology &operator=(const CNNTopology &) = default;
    CNNTopology &operator=(CNNTopology &&) = default;

    const_iterator cbegin() const { return layers.begin(); }
    const_iterator begin() const { return layers.begin(); }

    const_iterator cend() const { return layers.end(); }
    const_iterator end() const { return layers.end(); }

    const std::shared_ptr<CNNTopologyLayer> &operator()(size_t index) const;

    void setActivationFunction(const af::ActivationFunctionType act_function) {
      activationFunction = act_function;
    }
    const af::ActivationFunctionType getActivationFunction() const { return activationFunction; }

    const std::pair<size_t, size_t> &getInputSize() const { return inputSize; }
    const size_t getDeepth() const { return layers.size(); }
    const std::vector<std::shared_ptr<CNNTopologyLayer>> &getTopology() const { return layers; }

  private:
    void addConvolution(const std::pair<size_t, size_t> &inputSize, const size_t features,
                        const std::pair<size_t, size_t> &filterSize,
                        const af::ActivationFunctionType aFunction, const size_t stride,
                        const size_t padding);

    void addPooling(const std::pair<size_t, size_t> &inputSize, const PoolingType poolingType,
                    const std::pair<size_t, size_t> &poolSize, const size_t stride);

  private:
    std::pair<size_t, size_t> inputSize;
    std::vector<std::shared_ptr<CNNTopologyLayer>> layers;
    af::ActivationFunctionType activationFunction;
  };

  const CNNTopology stringToTopology(const std::string &str);

  std::ostream &operator<<(std::ostream &os, const CNNTopologyLayer &nn);
}   // namespace cnnet