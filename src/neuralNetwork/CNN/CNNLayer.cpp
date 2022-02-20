#include "CNNLayer.hpp"


namespace cnnet {

  CNNLayer::CNNLayer(const size_t stride) : stride(stride) {}


  CNNConvolutionLayer::CNNConvolutionLayer(std::pair<size_t, size_t> sizeFilter,
                                           const size_t stride, const size_t padding)
      : filter(sizeFilter), CNNLayer(stride), padding(padding) {}


  void CNNConvolutionLayer::compute(const FloatMatrix &input, FloatMatrix &output) {
    const FloatMatrix &matFiltre = filter.getMatrix();

    size_t rowsPos = 0;
    size_t colsPos = 0;

    for (size_t i = 0; i < output.getRows() - 2 * padding; i++) {
      for (size_t j = 0; j < output.getCols() - 2 * padding; j++) {
        float sum = 0.f;
        for (size_t k = 0; k < matFiltre.getRows(); k++) {
          for (size_t l = 0; l < matFiltre.getCols(); l++) {
            sum += input(k + rowsPos, l + colsPos) * matFiltre(k, l);
          }
        }
        output(i + padding, j + padding) = sum;
        colsPos += stride;
      }
      rowsPos += stride;
      colsPos = 0;
    }
  }


  void CNNConvolutionLayer::computeBackward(const FloatMatrix &input, CNNStorageBP &storage) {
    auto &convoStorage = static_cast<CNNStorageBPConvolution &>(storage);

    FloatMatrix dilatedOut4Intput(8, 8), dilatedOut4Filter(4, 4);

    for (auto &i : dilatedOut4Intput) { i = 0.f; }
    for (auto &i : dilatedOut4Filter) { i = 0.f; }
    fillDilatedMatrix(storage.output, dilatedOut4Intput, stride - 1,
                      std::make_pair(filter.getRows() - 1, filter.getCols() - 1));
    fillDilatedMatrix(storage.output, dilatedOut4Filter, stride - 1, std::make_pair(0, 0));

    std::cout << "dilated out for input: " << std::endl;
    std::cout << dilatedOut4Intput << std::endl;
    std::cout << "dilated out for filter: " << std::endl;
    std::cout << dilatedOut4Filter << std::endl;

    const FloatMatrix &matFiltre = filter.getMatrix();

    // compute input error
    size_t rowsPos = 0;
    size_t colsPos = 0;
    for (size_t i = 0; i < convoStorage.errorInput.getRows(); i++) {
      for (size_t j = 0; j < convoStorage.errorInput.getCols(); j++) {
        float sum = 0.f;
        for (long k = 0; k < filter.getRows(); k++) {
          for (long l = 0; l < filter.getCols(); l++) {
            sum += dilatedOut4Intput(k + rowsPos, l + colsPos) *
                   matFiltre(std::abs(k - (long) filter.getRows() + 1),
                             std::abs(l - (long) filter.getCols() + 1));
          }
        }
        convoStorage.errorInput(i, j) = sum;
        colsPos++;
      }
      rowsPos++;
      colsPos = 0;
    }


    // compute filter error
    rowsPos = 0;
    colsPos = 0;
    for (size_t i = 0; i < convoStorage.errorFilter.getRows(); i++) {
      for (size_t j = 0; j < convoStorage.errorFilter.getCols(); j++) {
        float sum = 0.f;
        for (long k = 0; k < dilatedOut4Filter.getRows(); k++) {
          for (long l = 0; l < dilatedOut4Filter.getCols(); l++) {
            sum += input(k + rowsPos, l + colsPos) * dilatedOut4Filter(k, l);
          }
        }
        convoStorage.errorFilter(i, j) = sum;
        colsPos++;
      }
      rowsPos++;
      colsPos = 0;
    }
  }


  CNNPoolingLayer::CNNPoolingLayer(const std::pair<size_t, size_t> poolSize, const size_t stride)
      : poolingSize(poolSize), CNNLayer(stride) {}


  CNNMaxPoolingLayer::CNNMaxPoolingLayer(const std::pair<size_t, size_t> poolSize,
                                         const size_t stride)
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


  CNNAvgPoolingLayer::CNNAvgPoolingLayer(const std::pair<size_t, size_t> poolSize,
                                         const size_t stride)
      : CNNPoolingLayer(poolSize, stride) {}

  void CNNAvgPoolingLayer::compute(const FloatMatrix &input, FloatMatrix &output) {
    const size_t incrMaxRows = ((input.getRows() - poolingSize.first) / stride) + 1;

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


  void fillDilatedMatrix(const FloatMatrix &input, FloatMatrix &dilated, const size_t stride,
                         const std::pair<size_t, size_t> padding) {
    for (size_t i = 0; i < input.getRows(); i++) {
      for (size_t j = 0; j < input.getCols(); j++) {
        dilated(i + padding.first + i * stride, j + padding.second + j * stride) = input(i, j);
      }
    }
  }

}   // namespace cnnet