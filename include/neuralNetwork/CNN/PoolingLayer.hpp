#pragma once

#include "Filter.hpp"

namespace cnnet {

  using namespace math;

  class PoolingLayer {
  public:
    PoolingLayer(const size_t size, const size_t stride);
    virtual ~PoolingLayer() = default;

    const size_t getStride() const { return stride; }

    virtual void compute(const FloatMatrix &input, FloatMatrix &output) = 0;

  protected:
    const size_t size;
    const size_t stride;
  };


  class MaxPoolingLayer : public PoolingLayer {
  public:
    MaxPoolingLayer(const size_t size, const size_t stride);
    ~MaxPoolingLayer() = default;

    void compute(const FloatMatrix &input, FloatMatrix &output) override;
  };


  class AvgPoolingLayer : public PoolingLayer {
  public:
    AvgPoolingLayer(const size_t size, const size_t stride);
    ~AvgPoolingLayer() = default;

    void compute(const FloatMatrix &input, FloatMatrix &output) override;
  };


}   // namespace cnnet