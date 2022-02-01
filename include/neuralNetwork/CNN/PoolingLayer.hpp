#pragma once

#include "Filter.hpp"
#include <utility>

namespace cnnet {

  using namespace math;

  class PoolingLayer {
  public:
    PoolingLayer(const std::pair<size_t, size_t> outputSize,
                 const std::pair<size_t, size_t> PoolSize, const size_t stride);
    virtual ~PoolingLayer() = default;

    const size_t getStride() const { return stride; }

    virtual const FloatMatrix &compute(const FloatMatrix &input) = 0;

  protected:
    FloatMatrix output;
    const std::pair<size_t, size_t> poolingSize;
    const size_t stride;
  };


  class MaxPoolingLayer : public PoolingLayer {
  public:
    MaxPoolingLayer(const std::pair<size_t, size_t> outputSize,
                    const std::pair<size_t, size_t> PoolSize, const size_t stride);
    ~MaxPoolingLayer() = default;

    const FloatMatrix &compute(const FloatMatrix &input) override;
  };


  class AvgPoolingLayer : public PoolingLayer {
  public:
    AvgPoolingLayer(const std::pair<size_t, size_t> outputSize,
                    const std::pair<size_t, size_t> PoolSize, const size_t stride);
    ~AvgPoolingLayer() = default;

    const FloatMatrix &compute(const FloatMatrix &input) override;
  };


}   // namespace cnnet