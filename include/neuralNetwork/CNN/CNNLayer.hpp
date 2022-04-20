#pragma once

#include "ActivationFunction.hpp"
#include "CNNStorageBP.hpp"
#include "Matrix.hpp"
#include "math/clFMatrix.hpp"
#include "math/clFTensor.hpp"
#include "openclUtils/clWrapper.hpp"
#include <iostream>


namespace nnet {
  using namespace math;

  enum class LayerType { CONVOLUTION, POOLING };
  enum class PoolingType { MAX, AVERAGE };

  class CNNLayer {
  public:
    explicit CNNLayer(const std::pair<size_t, size_t> outputSize);
    virtual ~CNNLayer() = default;

    virtual std::unique_ptr<CNNLayer> copy() const = 0;

    // Pretty messy but easiest way to do it for now
    [[nodiscard]] virtual bool hasWeight() const { return false; }

    virtual clFTensor &getWeight() {
      throw std::runtime_error("CNNLayer: Tried to acces weight in a layer without one ");
    }

    virtual const clFTensor &getWeight() const {
      throw std::runtime_error("CNNLayer: Tried to acces weight in a layer without one ");
    }

    virtual void setWeight(const clFTensor &weights);

    virtual clFTensor compute(const clFTensor &input) = 0;
    virtual clFTensor computeForward(const clFTensor &input, CNNStorageBP &storage) = 0;
    virtual clFTensor computeBackward(const clFTensor &input, CNNStorageBP &storage) = 0;

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

    clFTensor &getWeight() override { return filters; }
    const clFTensor &getWeight() const override { return filters; }
    void setWeight(const clFTensor &weights) override;

    [[nodiscard]] const clFTensor &getFilter() const { return filters; }
    [[nodiscard]] clFTensor &getFilter() { return filters; }

    clFTensor compute(const clFTensor &input) override;
    clFTensor computeForward(const clFTensor &input, CNNStorageBP &storage) override;
    clFTensor computeBackward(const clFTensor &input, CNNStorageBP &storage) override;

  private:
    const size_t n_branch;
    const size_t n_filter;

    clFTensor filters;
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

    clFTensor compute(const clFTensor &input) override;
    clFTensor computeForward(const clFTensor &input, CNNStorageBP &storage) override;
    clFTensor computeBackward(const clFTensor &input, CNNStorageBP &storage) override;
  };


  class CNNAvgPoolingLayer final : public CNNPoolingLayer {
  public:
    CNNAvgPoolingLayer(const std::pair<size_t, size_t> outputSize,
                       const std::pair<size_t, size_t> PoolSize);
    CNNAvgPoolingLayer(const CNNAvgPoolingLayer &other);
    ~CNNAvgPoolingLayer() override = default;

    std::unique_ptr<CNNLayer> copy() const override;

    clFTensor compute(const clFTensor &input) override;
    clFTensor computeForward(const clFTensor &input, CNNStorageBP &storage) override;
    clFTensor computeBackward(const clFTensor &input, CNNStorageBP &storage) override;

  private:
    clFTensor filter;
  };

}   // namespace nnet