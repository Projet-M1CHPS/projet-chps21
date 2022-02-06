#pragma once


#include <iostream>

#include "Filter.hpp"
#include "Matrix.hpp"

namespace cnnet {

  using namespace math;


  class CNNLayer {
  public:
    CNNLayer(const size_t stride);
    virtual ~CNNLayer() = default;

    const size_t getStride() const { return stride; };
    virtual void compute(const FloatMatrix &input, FloatMatrix &output) = 0;

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


  private:
    Filter filter;
    const size_t padding;
  };


  class CNNPoolingLayer : public CNNLayer {
  public:
    CNNPoolingLayer(const std::pair<size_t, size_t> PoolSize, const size_t stride);
    virtual ~CNNPoolingLayer() = default;

    virtual void compute(const FloatMatrix &input, FloatMatrix &output) = 0;

  protected:
    const std::pair<size_t, size_t> poolingSize;
  };


  class CNNMaxPoolingLayer : public CNNPoolingLayer {
  public:
    CNNMaxPoolingLayer(const std::pair<size_t, size_t> PoolSize, const size_t stride);
    ~CNNMaxPoolingLayer() = default;

    void compute(const FloatMatrix &input, FloatMatrix &output) override;
  };


  class CNNAvgPoolingLayer : public CNNPoolingLayer {
  public:
    CNNAvgPoolingLayer(const std::pair<size_t, size_t> PoolSize, const size_t stride);
    ~CNNAvgPoolingLayer() = default;

    void compute(const FloatMatrix &input, FloatMatrix &output) override;
  };


}   // namespace cnnet