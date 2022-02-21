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


  void CNNConvolutionLayer::computeForward(FloatMatrix &input, CNNStorageBP &storage) {
    compute(input, storage.output);
  }


  void CNNConvolutionLayer::computeBackward(FloatMatrix &input, CNNStorageBP &storage) {
    auto &convoStorage = static_cast<CNNStorageBPConvolution &>(storage);

    fillDilatedMatrix(convoStorage.output, convoStorage.dilated4Input, stride - 1,
                      std::make_pair(filter.getRows() - 1, filter.getCols() - 1));
    fillDilatedMatrix(convoStorage.output, convoStorage.dilated4Filter, stride - 1,
                      std::make_pair(0, 0));

    std::cout << "dilated out for input: " << std::endl;
    std::cout << convoStorage.dilated4Input << std::endl;
    std::cout << "dilated out for filter: " << std::endl;
    std::cout << convoStorage.dilated4Filter << std::endl;
    std::cout << "filter: " << std::endl;
    std::cout << filter.getMatrix() << std::endl;

    const FloatMatrix &matFiltre = filter.getMatrix();


    // compute filter error
    size_t rowsPos = 0;
    size_t colsPos = 0;
    for (size_t i = 0; i < convoStorage.errorFilter.getRows(); i++) {
      for (size_t j = 0; j < convoStorage.errorFilter.getCols(); j++) {
        float sum = 0.f;
        for (long k = 0; k < convoStorage.dilated4Filter.getRows(); k++) {
          for (long l = 0; l < convoStorage.dilated4Filter.getCols(); l++) {
            sum += input(k + rowsPos, l + colsPos) * convoStorage.dilated4Filter(k, l);
          }
        }
        convoStorage.errorFilter(i, j) = sum;
        colsPos++;
      }
      rowsPos++;
      colsPos = 0;
    }

    std::cout << "error filter: " << std::endl;
    std::cout << convoStorage.errorFilter << std::endl;


    // compute input error
    rowsPos = 0;
    colsPos = 0;
    for (size_t i = 0; i < input.getRows(); i++) {
      for (size_t j = 0; j < input.getCols(); j++) {
        float sum = 0.f;
        for (long k = 0; k < filter.getRows(); k++) {
          for (long l = 0; l < filter.getCols(); l++) {
            sum += convoStorage.dilated4Input(k + rowsPos, l + colsPos) *
                   matFiltre(std::abs(k - (long) filter.getRows() + 1),
                             std::abs(l - (long) filter.getCols() + 1));
          }
        }
        input(i, j) = sum;
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
    size_t rowsPos = 0, colsPos = 0;

    for (size_t i = 0; i < output.getRows(); i++) {
      for (size_t j = 0; j < output.getCols(); j++) {
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

  void CNNMaxPoolingLayer::computeForward(FloatMatrix &input, CNNStorageBP &storage) {
    auto &poolingStorage = static_cast<CNNStorageBPMaxPooling &>(storage);
    size_t rowsPos = 0, colsPos = 0;

    for (size_t i = 0; i < poolingStorage.output.getRows(); i++) {
      for (size_t j = 0; j < poolingStorage.output.getCols(); j++) {
        float max = input(rowsPos, colsPos);
        poolingStorage.maxIndex(i, j).first = rowsPos;
        poolingStorage.maxIndex(i, j).second = colsPos;
        for (size_t k = 0; k < poolingSize.first; k++) {
          for (size_t l = 0; l < poolingSize.second; l++) {
            if (input(k + rowsPos, l + colsPos) > max) {
              max = input(k + rowsPos, l + colsPos);
              poolingStorage.maxIndex(i, j).first = k + rowsPos;
              poolingStorage.maxIndex(i, j).second = l + colsPos;
            }
          }
        }
        poolingStorage.output(i, j) = max;
        colsPos += stride;
      }
      colsPos = 0;
      rowsPos += stride;
    }
  }

  void CNNMaxPoolingLayer::computeBackward(FloatMatrix &input, CNNStorageBP &storage) {
    auto &poolingStorage = static_cast<CNNStorageBPMaxPooling &>(storage);
    std::cout << "compute backprop max pooling" << std::endl;

    for (auto &i : input) { i = 0.f; }

    for (size_t i = 0; i < poolingStorage.output.getRows(); i++) {
      for (size_t j = 0; j < poolingStorage.output.getCols(); j++) {
        input(poolingStorage.maxIndex(i, j).first, poolingStorage.maxIndex(i, j).second) +=
                poolingStorage.output(i, j);
      }
    }
  }

  CNNAvgPoolingLayer::CNNAvgPoolingLayer(const std::pair<size_t, size_t> poolSize,
                                         const size_t stride)
      : CNNPoolingLayer(poolSize, stride) {}

  void CNNAvgPoolingLayer::compute(const FloatMatrix &input, FloatMatrix &output) {
    size_t rowsPos = 0, colsPos = 0;

    for (size_t i = 0; i < output.getRows(); i++) {
      for (size_t j = 0; j < output.getCols(); j++) {
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

  void CNNAvgPoolingLayer::computeForward(FloatMatrix &input, CNNStorageBP &storage) {
    compute(input, storage.output);
  }

  void CNNAvgPoolingLayer::computeBackward(FloatMatrix &input, CNNStorageBP &storage) {
    std::cout << "compute backprop max pooling" << std::endl;
    auto &poolingStorage = static_cast<CNNStorageBPMaxPooling &>(storage);

    std::cout << storage.output << std::endl;
    for (auto &i : input) { i = 0.f; }

    size_t rowsPos = 0, colsPos = 0;
    for (size_t i = 0; i < poolingStorage.output.getRows(); i++) {
      for (size_t j = 0; j < poolingStorage.output.getCols(); j++) {
        const float factor = 1.f / (float) (poolingSize.first * poolingSize.second);
        for (size_t k = 0; k < poolingSize.first; k++) {
          for (size_t l = 0; l < poolingSize.second; l++) {
            input(k + rowsPos, l + colsPos) += poolingStorage.output(i, j) * factor;
          }
        }
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