#pragma once

#include "ActivationFunction.hpp"
#include "CNNStorageBP.hpp"
#include "Filter.hpp"
#include "Matrix.hpp"
#include "clUtils/clFMatrix.hpp"
#include "clUtils/clWrapper.hpp"
#include <iostream>


namespace nnet {
  using namespace math;

  enum class LayerType { CONVOLUTION, POOLING };
  enum class PoolingType { MAX, AVERAGE };

  class CNNLayer {
  public:
    explicit CNNLayer(const size_t stride);

    [[nodiscard]] const size_t getStride() const { return stride; };
    virtual void compute(const clFMatrix &input) = 0;
    virtual void computeForward(const clFMatrix &input, CNNStorageBP &storage) = 0;
    virtual void computeBackward(const clFMatrix &input, CNNStorageBP &storage) = 0;

    [[nodiscard]] virtual const math::clFMatrix &getOutput(const size_t index) const = 0;

  protected:
    const size_t stride;
  };


  class CNNConvolutionLayer : public CNNLayer {
  public:
    CNNConvolutionLayer(const std::pair<size_t, size_t> outputSize,
                        const std::pair<size_t, size_t> sizeFilter,
                        const af::ActivationFunctionType aFunction, const size_t stride,
                        const size_t padding = 0);

    [[nodiscard]] const size_t getPadding() const { return padding; };
    [[nodiscard]] const Filter &getFilter() const { return filter; }

    void compute(const clFMatrix &input) override;
    void computeForward(const clFMatrix &input, CNNStorageBP &storage) override;
    void computeBackward(const clFMatrix &input, CNNStorageBP &storage) override;

    [[nodiscard]] const math::clFMatrix &getOutput(const size_t index) const override {
      return output;
    };

  private:
    const size_t padding;
    af::ActivationFunctionType activationFunction;

    // TODO : Replace filter class with tensor of filters and tensor of output
    Filter filter;
    clFMatrix output;
  };


  class CNNPoolingLayer : public CNNLayer {
  public:
    CNNPoolingLayer(const std::pair<size_t, size_t> outputSize,
                    const std::pair<size_t, size_t> PoolSize, const size_t stride);

    [[nodiscard]] const math::clFMatrix &getOutput(const size_t index) const override {
      return output;
    };

  protected:
    const std::pair<size_t, size_t> poolingSize;
    clFMatrix output;
  };


  class CNNMaxPoolingLayer : public CNNPoolingLayer {
  public:
    CNNMaxPoolingLayer(const std::pair<size_t, size_t> outputSize,
                       const std::pair<size_t, size_t> PoolSize, const size_t stride);

    void compute(const clFMatrix &input) override;
    void computeForward(const clFMatrix &input, CNNStorageBP &storage) override;
    void computeBackward(const clFMatrix &input, CNNStorageBP &storage) override;
  };


  class CNNAvgPoolingLayer : public CNNPoolingLayer {
  public:
    CNNAvgPoolingLayer(const std::pair<size_t, size_t> outputSize,
                       const std::pair<size_t, size_t> PoolSize, const size_t stride);

    void compute(const clFMatrix &input) override;
    void computeForward(const clFMatrix &input, CNNStorageBP &storage) override;
    void computeBackward(const clFMatrix &input, CNNStorageBP &storage) override;
  };


  void fillDilatedMatrix(const FloatMatrix &input, FloatMatrix &dilated, const size_t stride,
                         const std::pair<size_t, size_t> padding);

}   // namespace nnet