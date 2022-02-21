#pragma once


#include <iostream>

#include "CNNStorageBP.hpp"
#include "Filter.hpp"
#include "Matrix.hpp"

namespace cnnet {

  using namespace math;


  enum class LayerType {
    CONVOLUTION,
    POOLING
  };

  enum class PoolingType {
    MAX,
    AVERAGE
  };



  class CNNLayer {
  public:
    CNNLayer(const size_t stride);
    virtual ~CNNLayer() = default;

    const size_t getStride() const { return stride; };
    virtual void compute(const FloatMatrix &input, FloatMatrix &output) = 0;
    virtual void computeForward(FloatMatrix &input, CNNStorageBP &storage) = 0;
    virtual void computeBackward(FloatMatrix &input, CNNStorageBP &storage) = 0;

  protected:
    const size_t stride;
  };


  class CNNConvolutionLayer : public CNNLayer {
  public:
    CNNConvolutionLayer(std::pair<size_t, size_t> sizeFilter, const size_t stride,
                        const size_t padding = 0);

    ~CNNConvolutionLayer() = default;

    const size_t getStride() const { return stride; };
    const size_t getPadding() const { return padding; };

    const Filter &getFilter() const { return filter; }


    void compute(const FloatMatrix &input, FloatMatrix &output) override;
    void computeForward(FloatMatrix &input, CNNStorageBP &storage) override;
    void computeBackward(FloatMatrix &input, CNNStorageBP &storage) override;


  private:
    Filter filter;
    const size_t padding;
  };


  class CNNPoolingLayer : public CNNLayer {
  public:
    CNNPoolingLayer(const std::pair<size_t, size_t> PoolSize, const size_t stride);
    virtual ~CNNPoolingLayer() = default;

    virtual void compute(const FloatMatrix &input, FloatMatrix &output) = 0;
    void computeForward(FloatMatrix &input, CNNStorageBP &storage) = 0;
    void computeBackward(FloatMatrix &input, CNNStorageBP &storage) = 0;

  protected:
    const std::pair<size_t, size_t> poolingSize;
  };


  class CNNMaxPoolingLayer : public CNNPoolingLayer {
  public:
    CNNMaxPoolingLayer(const std::pair<size_t, size_t> PoolSize, const size_t stride);
    ~CNNMaxPoolingLayer() = default;

    void compute(const FloatMatrix &input, FloatMatrix &output) override;
    void computeForward(FloatMatrix &input, CNNStorageBP &storage) override;
    void computeBackward(FloatMatrix &input, CNNStorageBP &storage) override;
  };


  class CNNAvgPoolingLayer : public CNNPoolingLayer {
  public:
    CNNAvgPoolingLayer(const std::pair<size_t, size_t> PoolSize, const size_t stride);
    ~CNNAvgPoolingLayer() = default;

    void compute(const FloatMatrix &input, FloatMatrix &output) override;
    void computeForward(FloatMatrix &input, CNNStorageBP &storage) override;
    void computeBackward(FloatMatrix &input, CNNStorageBP &storage) override;
  };


  void fillDilatedMatrix(const FloatMatrix &input, FloatMatrix &dilated, const size_t stride,
                         const std::pair<size_t, size_t> padding);

}   // namespace cnnet