#include "ConvolutionLayer.hpp"

namespace cnnet {


  ConvolutionLayer::ConvolutionLayer(const size_t rowsFiltre, const size_t colsFiltre,
                                     const size_t stride, const size_t padding)
      : filter(rowsFiltre, colsFiltre), stride(stride), padding(padding) {}


  ConvolutionLayer::ConvolutionLayer(std::pair<size_t, size_t> sizeFilter, const size_t stride,
                                     const size_t padding)
      : filter(sizeFilter), stride(stride), padding(padding) {}


  void ConvolutionLayer::compute(const FloatMatrix &input, const FloatMatrix &output) {
    // deplacement du filtre sur la feature
  }
}   // namespace cnnet