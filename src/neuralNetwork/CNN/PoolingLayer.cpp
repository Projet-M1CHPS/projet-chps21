#include "PoolingLayer.hpp"


namespace cnnet {

  PoolingLayer::PoolingLayer(const std::pair<size_t, size_t> outputSize,
                             const std::pair<size_t, size_t> poolSize, const size_t stride)
      : output(outputSize.first, outputSize.second), poolingSize(poolSize), stride(stride) {}


  MaxPoolingLayer::MaxPoolingLayer(const std::pair<size_t, size_t> outputSize,
                                   const std::pair<size_t, size_t> poolSize, const size_t stride)
      : PoolingLayer(outputSize, poolSize, stride) {}

  const FloatMatrix &MaxPoolingLayer::compute(const FloatMatrix &input) {
    const size_t incrMaxRows = ((input.getRows() - poolingSize.first) / stride) + 1;   //* nombre de layer;

    size_t rowsPos = 0, colsPos = 0;

    for (size_t i = 0; i < incrMaxRows; i++) {
      for (size_t j = 0; j < incrMaxRows; j++) {
        float max = input(rowsPos, colsPos);
        for (size_t k = 0; k < poolingSize.first; k++) {
          for (size_t l = 0; l < poolingSize.second; l++) {
            max = std::max(max, input(k + rowsPos, l + colsPos));
          }
        }
        output(i, j) = max;
        colsPos += stride;
      }
      colsPos = 0;
      rowsPos += stride;
    }

    return output;
  }


  AvgPoolingLayer::AvgPoolingLayer(const std::pair<size_t, size_t> outputSize,
                                   const std::pair<size_t, size_t> poolSize, const size_t stride)
      : PoolingLayer(outputSize, poolSize, stride) {}

  const FloatMatrix &AvgPoolingLayer::compute(const FloatMatrix &input) {
    const size_t incrMaxRows = ((input.getRows() - poolingSize.first) / stride) + 1;   //* nombre de layer;

    size_t rowsPos = 0, colsPos = 0;

    for (size_t i = 0; i < incrMaxRows; i++) {
      for (size_t j = 0; j < incrMaxRows; j++) {
        float sum = 0.f;
        for (size_t k = 0; k < poolingSize.first; k++) {
          for (size_t l = 0; l < poolingSize.second; l++) { sum += input(k + rowsPos, l + colsPos); }
        }
        output(i, j) = sum / (poolingSize.first * poolingSize.second);
        colsPos += stride;
      }
      colsPos = 0;
      rowsPos += stride;
    }

    return output;
  }

}   // namespace cnnet