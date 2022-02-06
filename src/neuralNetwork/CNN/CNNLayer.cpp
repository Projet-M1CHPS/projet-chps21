#include "CNNLayer.hpp"


namespace cnnet {

  CNNLayer::CNNLayer(const size_t stride) : stride(stride) {}



  CNNConvolutionLayer::CNNConvolutionLayer(std::pair<size_t, size_t> sizeFilter, const size_t stride,
                                     const size_t padding)
      : filter(sizeFilter), CNNLayer(stride), padding(padding) {}


  void CNNConvolutionLayer::compute(const FloatMatrix &input, FloatMatrix &output) {
    const FloatMatrix &matFiltre = filter.getMatrix();

    int rowsPos = 0;
    int colsPos = 0;

    for (size_t i = 0; i < output.getRows() - 2 * padding; i++) {
      for (size_t j = 0; j < output.getCols() - 2 * padding; j++) {
        float sum = 0.f;
        // std::cout << "sum = 0" << std::endl;
        for (size_t k = 0; k < matFiltre.getRows(); k++) {
          for (size_t l = 0; l < matFiltre.getCols(); l++) {
            sum += input(k + rowsPos, l + colsPos) * matFiltre(k, l);
          }
        }
        std::cout << "index " << i + padding << " " << j + padding << " " << sum << std::endl;
        output(i + padding, j + padding) = sum;
        colsPos += stride;
      }
      rowsPos += stride;
      colsPos = 0;
    }
  }


  CNNPoolingLayer::CNNPoolingLayer(const std::pair<size_t, size_t> poolSize, const size_t stride)
      : poolingSize(poolSize), CNNLayer(stride) {}


  CNNMaxPoolingLayer::CNNMaxPoolingLayer(const std::pair<size_t, size_t> poolSize, const size_t stride)
      : CNNPoolingLayer(poolSize, stride) {}

  void CNNMaxPoolingLayer::compute(const FloatMatrix &input, FloatMatrix &output) {
    const size_t incrMaxRows =
            ((input.getRows() - poolingSize.first) / stride) + 1;   //* nombre de layer;

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
  }


  CNNAvgPoolingLayer::CNNAvgPoolingLayer(const std::pair<size_t, size_t> poolSize, const size_t stride)
      : CNNPoolingLayer(poolSize, stride) {}

  void CNNAvgPoolingLayer::compute(const FloatMatrix &input, FloatMatrix &output) {
    const size_t incrMaxRows =
            ((input.getRows() - poolingSize.first) / stride) + 1;   //* nombre de layer;

    size_t rowsPos = 0, colsPos = 0;

    for (size_t i = 0; i < incrMaxRows; i++) {
      for (size_t j = 0; j < incrMaxRows; j++) {
        float sum = 0.f;
        for (size_t k = 0; k < poolingSize.first; k++) {
          for (size_t l = 0; l < poolingSize.second; l++) {
            sum += input(k + rowsPos, l + colsPos);
          }
        }
        output(i, j) = sum / (poolingSize.first * poolingSize.second);
        colsPos += stride;
      }
      colsPos = 0;
      rowsPos += stride;
    }
  }

}   // namespace cnnet