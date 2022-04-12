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

    virtual clFTensor compute(const clFTensor &input) = 0;
    virtual clFTensor computeForward(const clFTensor &input, CNNStorageBP &storage) = 0;
    virtual clFTensor computeBackward(const clFTensor &input, CNNStorageBP &storage) = 0;

  protected:
    const std::pair<size_t, size_t> outputSize;
  };


  class CNNConvolutionLayer final : public CNNLayer {
  public:
    // nFilter nombre de kernel par branche
    CNNConvolutionLayer(const std::pair<size_t, size_t> outputSize,
                        const std::pair<size_t, size_t> sizeFilter, const size_t nFilter,
                        const af::ActivationFunctionType aFunction,
                        const size_t nBranch);

    [[nodiscard]] const clFTensor &getFilter() const { return filters; }
    [[nodiscard]] clFTensor &getFilter() { return filters; }

    clFTensor compute(const clFTensor &input) override;
    clFTensor computeForward(const clFTensor &input, CNNStorageBP &storage) override;
    clFTensor computeBackward(const clFTensor &input, CNNStorageBP &storage) override;

  private:
    const size_t n_branch;
    const size_t n_filter;
    af::ActivationFunctionType activationFunction;

    clFTensor filters;
  };


  class CNNPoolingLayer : public CNNLayer {
  public:
    CNNPoolingLayer(const std::pair<size_t, size_t> outputSize,
                    const std::pair<size_t, size_t> PoolSize);

  protected:
    const std::pair<size_t, size_t> poolingSize;
  };


  class CNNMaxPoolingLayer final : public CNNPoolingLayer {
  public:
    CNNMaxPoolingLayer(const std::pair<size_t, size_t> outputSize,
                       const std::pair<size_t, size_t> PoolSize);

    clFTensor compute(const clFTensor &input) override;
    clFTensor computeForward(const clFTensor &input, CNNStorageBP &storage) override;
    clFTensor computeBackward(const clFTensor &input, CNNStorageBP &storage) override;
  };


  class CNNAvgPoolingLayer final : public CNNPoolingLayer {
  public:
    CNNAvgPoolingLayer(const std::pair<size_t, size_t> outputSize,
                       const std::pair<size_t, size_t> PoolSize);

    clFTensor compute(const clFTensor &input) override;
    clFTensor computeForward(const clFTensor &input, CNNStorageBP &storage) override;
    clFTensor computeBackward(const clFTensor &input, CNNStorageBP &storage) override;
  };

}   // namespace nnet