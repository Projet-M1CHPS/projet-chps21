#pragma once

#include "ActivationFunction.hpp"
#include "CNNStorageBP.hpp"
#include "Filter.hpp"
#include "math/Matrix.hpp"
#include "math/clFMatrix.hpp"
#include "openclUtils/clWrapper.hpp"
#include <iostream>


namespace nnet {
  using namespace math;

  enum class LayerType { CONVOLUTION, POOLING };
  enum class PoolingType { MAX, AVERAGE };

  class CNNLayer {
  public:
    explicit CNNLayer(const std::pair<size_t, size_t> outputSize, const size_t stride);

    [[nodiscard]] const size_t getStride() const { return stride; };
    virtual clFMatrix compute(const clFMatrix &input) = 0;
    virtual void computeForward(const clFMatrix &input, CNNStorageBP &storage) = 0;
    virtual void computeBackward(const clFMatrix &input, CNNStorageBP &storage) = 0;

  protected:
    const size_t stride;
    const std::pair<size_t, size_t> outputSize;
  };


  class CNNConvolutionLayer : public CNNLayer {
  public:
    CNNConvolutionLayer(const std::pair<size_t, size_t> outputSize,
                        const std::pair<size_t, size_t> sizeFilter,
                        const af::ActivationFunctionType aFunction, const size_t stride,
                        const size_t padding = 0);

    [[nodiscard]] const size_t getPadding() const { return padding; };
    [[nodiscard]] const Filter &getFilter() const { return filter; }

    clFMatrix compute(const clFMatrix &input) override;
    void computeForward(const clFMatrix &input, CNNStorageBP &storage) override;
    void computeBackward(const clFMatrix &input, CNNStorageBP &storage) override;

  private:
    const size_t padding;
    af::ActivationFunctionType activationFunction;

    // TODO : Replace filter class with tensor of filters and tensor of output
    Filter filter;
  };


  class CNNPoolingLayer : public CNNLayer {
  public:
    CNNPoolingLayer(const std::pair<size_t, size_t> outputSize,
                    const std::pair<size_t, size_t> PoolSize, const size_t stride);

  protected:
    const std::pair<size_t, size_t> poolingSize;
  };


  class CNNMaxPoolingLayer : public CNNPoolingLayer {
  public:
    CNNMaxPoolingLayer(const std::pair<size_t, size_t> outputSize,
                       const std::pair<size_t, size_t> PoolSize, const size_t stride);

    clFMatrix compute(const clFMatrix &input) override;
    void computeForward(const clFMatrix &input, CNNStorageBP &storage) override;
    void computeBackward(const clFMatrix &input, CNNStorageBP &storage) override;
  };


  class CNNAvgPoolingLayer : public CNNPoolingLayer {
  public:
    CNNAvgPoolingLayer(const std::pair<size_t, size_t> outputSize,
                       const std::pair<size_t, size_t> PoolSize, const size_t stride);

    clFMatrix compute(const clFMatrix &input) override;
    void computeForward(const clFMatrix &input, CNNStorageBP &storage) override;
    void computeBackward(const clFMatrix &input, CNNStorageBP &storage) override;
  };


  void fillDilatedMatrix(const FloatMatrix &input, FloatMatrix &dilated, const size_t stride,
                         const std::pair<size_t, size_t> padding);

}   // namespace nnet