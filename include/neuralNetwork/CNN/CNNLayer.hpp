#pragma once

#include "ActivationFunction.hpp"
#include "CNNStorageBP.hpp"
#include "Matrix.hpp"
#include "math/clFMatrix.hpp"
#include "math/clFTensor.hpp"
#include "openclUtils/clWrapper.hpp"
#include <iostream>


namespace nnet {
  enum class LayerType { CONVOLUTION, POOLING };
  enum class PoolingType { MAX, AVERAGE };

  class CNNLayer {
  public:
    explicit CNNLayer(const std::pair<size_t, size_t> outputSize);
    virtual ~CNNLayer() = default;

    virtual std::unique_ptr<CNNLayer> copy() const = 0;

    // Pretty messy but easiest way to do it for now
    [[nodiscard]] virtual bool hasWeight() const { return false; }

    virtual math::clFTensor &getWeight() {
      throw std::runtime_error("CNNLayer: Tried to acces weight in a layer without one ");
    }

    virtual const math::clFTensor &getWeight() const {
      throw std::runtime_error("CNNLayer: Tried to acces weight in a layer without one ");
    }

    virtual void setWeight(const math::clFTensor &weights);

    virtual math::clFTensor compute(const math::clFTensor &input) = 0;
    virtual math::clFTensor computeForward(const math::clFTensor &input, CNNStorageBP &storage) = 0;
    virtual math::clFTensor computeBackward(const math::clFTensor &input, CNNStorageBP &storage) = 0;

    [[nodiscard]] const std::pair<size_t, size_t> getOutputSize() const { return outputSize; }

  protected:
    const std::pair<size_t, size_t> outputSize;
  };


  class CNNConvolutionLayer final : public CNNLayer {
  public:
    // nFilter nombre de kernel par branche
    CNNConvolutionLayer(const std::pair<size_t, size_t> outputSize,
                        const std::pair<size_t, size_t> sizeFilter, const size_t nFilter,
                        const af::ActivationFunctionType aFunction, const size_t nBranch);

    CNNConvolutionLayer(const CNNConvolutionLayer &other);
    ~CNNConvolutionLayer() override = default;

    [[nodiscard]] std::unique_ptr<CNNLayer> copy() const override;

    // Pretty messy but easiest way to do it for now
    bool hasWeight() const override { return true; }

    math::clFTensor &getWeight() override { return filters; }
    //TODO : hum interressant plus de function pour faire la meme chose ?
    const math::clFTensor &getWeight() const override { return filters; }
    void setWeight(const math::clFTensor &weights) override;

    [[nodiscard]] const math::clFTensor &getFilter() const { return filters; }
    [[nodiscard]] math::clFTensor &getFilter() { return filters; }

    math::clFTensor compute(const math::clFTensor &input) override;
    math::clFTensor computeForward(const math::clFTensor &input, CNNStorageBP &storage) override;
    math::clFTensor computeBackward(const math::clFTensor &input, CNNStorageBP &storage) override;

  private:
    const size_t n_branch;
    const size_t n_filter;

    math::clFTensor filters;
    af::ActivationFunctionType a_function;
  };


  class CNNPoolingLayer : public CNNLayer {
  public:
    CNNPoolingLayer(const std::pair<size_t, size_t> outputSize,
                    const std::pair<size_t, size_t> PoolSize);
    ~CNNPoolingLayer() override = default;

  protected:
    const std::pair<size_t, size_t> poolingSize;
  };


  class CNNMaxPoolingLayer final : public CNNPoolingLayer {
  public:
    CNNMaxPoolingLayer(const std::pair<size_t, size_t> outputSize,
                       const std::pair<size_t, size_t> PoolSize);
    CNNMaxPoolingLayer(const CNNMaxPoolingLayer &other);
    ~CNNMaxPoolingLayer() override = default;

    std::unique_ptr<CNNLayer> copy() const override;

    math::clFTensor compute(const math::clFTensor &input) override;
    math::clFTensor computeForward(const math::clFTensor &input, CNNStorageBP &storage) override;
    math::clFTensor computeBackward(const math::clFTensor &input, CNNStorageBP &storage) override;
  };


  class CNNAvgPoolingLayer final : public CNNPoolingLayer {
  public:
    CNNAvgPoolingLayer(const std::pair<size_t, size_t> outputSize,
                       const std::pair<size_t, size_t> PoolSize);
    CNNAvgPoolingLayer(const CNNAvgPoolingLayer &other);
    ~CNNAvgPoolingLayer() override = default;

    std::unique_ptr<CNNLayer> copy() const override;

    math::clFTensor compute(const math::clFTensor &input) override;
    math::clFTensor computeForward(const math::clFTensor &input, CNNStorageBP &storage) override;
    math::clFTensor computeBackward(const math::clFTensor &input, CNNStorageBP &storage) override;

  private:
    math::clFTensor filter;
  };

}   // namespace nnet