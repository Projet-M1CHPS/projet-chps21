#pragma once

#include "ActivationFunction.hpp"
#include "CNNStorageBP.hpp"
#include "Matrix.hpp"
#include "clUtils/clFMatrix.hpp"
#include "clUtils/clFTensor.hpp"
#include "clUtils/clWrapper.hpp"
#include <iostream>


namespace nnet {
  using namespace math;

  enum class LayerType { CONVOLUTION, POOLING };
  enum class PoolingType { MAX, AVERAGE };

  class CNNLayer {
  public:
    explicit CNNLayer(const std::pair<size_t, size_t> outputSize, const size_t stride);

    [[nodiscard]] const size_t getStride() const { return stride; };
    virtual clFTensor compute(const clFTensor &input) = 0;
    virtual clFTensor computeForward(const clFTensor &input, CNNStorageBP &storage) = 0;
    virtual clFTensor computeBackward(const clFTensor &input, CNNStorageBP &storage) = 0;

  protected:
    const size_t stride;
    const std::pair<size_t, size_t> outputSize;
  };


  class CNNConvolutionLayer final : public CNNLayer {
  public:
    // nFilter nombre de kernel par branche
    CNNConvolutionLayer(const std::pair<size_t, size_t> outputSize,
                        const std::pair<size_t, size_t> sizeFilter, const size_t nFilter,
                        const af::ActivationFunctionType aFunction, const size_t stride,
                        const size_t nBranch, const size_t padding = 0);

    [[nodiscard]] const size_t getPadding() const { return padding; };
    [[nodiscard]] const clFTensor &getFilter() const { return filters; }
    [[nodiscard]] clFTensor &getFilter() { return filters; }

    clFTensor compute(const clFTensor &input) override;
    clFTensor computeForward(const clFTensor &input, CNNStorageBP &storage) override;
    clFTensor computeBackward(const clFTensor &input, CNNStorageBP &storage) override;

  private:
    const size_t n_branch;
    const size_t n_filter;
    const size_t padding;
    af::ActivationFunctionType activationFunction;

    clFTensor filters;
  };


  class CNNPoolingLayer : public CNNLayer {
  public:
    CNNPoolingLayer(const std::pair<size_t, size_t> outputSize,
                    const std::pair<size_t, size_t> PoolSize, const size_t stride);

  protected:
    const std::pair<size_t, size_t> poolingSize;
  };


  class CNNMaxPoolingLayer final : public CNNPoolingLayer {
  public:
    CNNMaxPoolingLayer(const std::pair<size_t, size_t> outputSize,
                       const std::pair<size_t, size_t> PoolSize, const size_t stride);

    clFTensor compute(const clFTensor &input) override;
    clFTensor computeForward(const clFTensor &input, CNNStorageBP &storage) override;
    clFTensor computeBackward(const clFTensor &input, CNNStorageBP &storage) override;
  };


  class CNNAvgPoolingLayer final : public CNNPoolingLayer {
  public:
    CNNAvgPoolingLayer(const std::pair<size_t, size_t> outputSize,
                       const std::pair<size_t, size_t> PoolSize, const size_t stride);

    clFTensor compute(const clFTensor &input) override;
    clFTensor computeForward(const clFTensor &input, CNNStorageBP &storage) override;
    clFTensor computeBackward(const clFTensor &input, CNNStorageBP &storage) override;
  };


  void fillDilatedMatrix(const FloatMatrix &input, FloatMatrix &dilated, const size_t stride,
                         const std::pair<size_t, size_t> padding);

}   // namespace nnet