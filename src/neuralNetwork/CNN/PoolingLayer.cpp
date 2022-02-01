#include "PoolingLayer.hpp"


namespace cnnet {

  PoolingLayer::PoolingLayer(const size_t size, const size_t stride) : size(size), stride(stride) {}


  MaxPoolingLayer::MaxPoolingLayer(const size_t size, const size_t stride)
      : PoolingLayer(size, stride) {}

  void MaxPoolingLayer::compute(const FloatMatrix &input, FloatMatrix &output) {
    const size_t incrMax = ((input.getRows() - size) / stride) + 1;   //* nombre de layer;

    size_t rowsPos = 0, colsPos = 0;

    for (size_t i = 0; i < incrMax; i++) {
      for (size_t j = 0; j < incrMax; j++) {
        float max = input(rowsPos, colsPos);
        for (size_t k = 0; k < size; k++) {
          for (size_t l = 0; l < size; l++) {
            max = std::max(max, input(k + rowsPos, l + colsPos));
          }
        }
        output(i, j) = max;
        colsPos += stride;
      }
      colsPos = 0;
      rowsPos += stride;
    }
  }


  AvgPoolingLayer::AvgPoolingLayer(const size_t size, const size_t stride)
      : PoolingLayer(size, stride) {}

  void AvgPoolingLayer::compute(const FloatMatrix &input, FloatMatrix &output) {
    const size_t incrMax = ((input.getRows() - size) / stride) + 1;   //* nombre de layer;

    size_t rowsPos = 0, colsPos = 0;

    for (size_t i = 0; i < incrMax; i++) {
      for (size_t j = 0; j < incrMax; j++) {
        float sum = 0.f;
        for (size_t k = 0; k < size; k++) {
          for (size_t l = 0; l < size; l++) {
            sum += input(k + rowsPos, l + colsPos);
          }
        }
        output(i, j) = sum / (size * size);
        colsPos += stride;
      }
      colsPos = 0;
      rowsPos += stride;
    }
  }

}   // namespace cnnet