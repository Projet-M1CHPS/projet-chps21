#pragma once

#include "Filter.hpp"

namespace cnnet {

  using namespace math;

  class PoolingLayer {
  public:
    PoolingLayer(const size_t stride, const size_t padding = 0);

    virtual ~PoolingLayer() = default;

    const size_t getStride() const { return stride; }
    const size_t getPadding() const { return padding; }

    void compute(const FloatMatrix &input, const FloatMatrix &output);

  private:
    virtual const float poolingMethode() = 0;

  private:
    const size_t stride, padding;
  };


  class MaxPoolingLayer : public PoolingLayer {
  public:
    MaxPoolingLayer(const size_t stride, const size_t padding = 0);
    ~MaxPoolingLayer() = default;

  private:
    const float poolingMethode() override;
  };


  class AvgPoolingLayer : public PoolingLayer {
  public:
    AvgPoolingLayer(const size_t stride, const size_t padding = 0);
    ~AvgPoolingLayer() = default;

  private:
    const float poolingMethode() override;
  };


}   // namespace cnnet