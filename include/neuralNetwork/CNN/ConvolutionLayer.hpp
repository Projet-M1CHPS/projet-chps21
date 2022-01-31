#pragma once

#include "Filter.hpp"
#include <utility>

namespace cnnet {

  using namespace math;

  class ConvolutionLayer {
  public:
    ConvolutionLayer(const size_t rowsF, const size_t colsF, const size_t stride,
                     const size_t padding = 0);
    ConvolutionLayer(std::pair<size_t, size_t> sizeFilter, const size_t stride,
                     const size_t padding = 0);

    ~ConvolutionLayer() = default;

    const size_t getStride() const { return stride; };
    const size_t getPadding() const { return padding; };

    const Filter &getFilter() const { return filter; }


    void compute(const FloatMatrix &input, FloatMatrix &output);


  private:
    Filter filter;
    const size_t stride, padding;
  };

}   // namespace cnnet