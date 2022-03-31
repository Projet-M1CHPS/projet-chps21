#include "CNNLayer.hpp"


namespace nnet {

  CNNLayer::CNNLayer(const size_t stride) : stride(stride) {}


  CNNConvolutionLayer::CNNConvolutionLayer(const std::pair<size_t, size_t> sizeFilter,
                                           const af::ActivationFunctionType aFunction,
                                           const size_t stride, const size_t padding)
      : filter(sizeFilter), CNNLayer(stride), padding(padding), activationFunction(aFunction) {
    // filter.randomize(0.f, 1.f);
  }


  void CNNConvolutionLayer::compute(const clFMatrix &_input, clFMatrix &_output) {
    FloatMatrix input = _input.toFloatMatrix(true);
    FloatMatrix output = _output.toFloatMatrix(true);

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

    auto afunc = af::getAFFromType(activationFunction).first;
    std::transform(output.cbegin(), output.cend(), output.begin(), afunc);

    _output = output;
  }


  void CNNConvolutionLayer::computeForward(const clFMatrix &input, CNNStorageBP &storage) {
    clFMatrix tmp = storage.output;
    compute(input, tmp);
    storage.output = tmp.toFloatMatrix(true);
  }


  void CNNConvolutionLayer::computeBackward(const clFMatrix &_input, CNNStorageBP &storage) {
    FloatMatrix input = _input.toFloatMatrix(true);

    auto &convoStorage = static_cast<CNNStorageBPConvolution &>(storage);

    fillDilatedMatrix(convoStorage.output, convoStorage.dilated4Input, stride - 1,
                      std::make_pair(filter.getRows() - 1, filter.getCols() - 1));
    fillDilatedMatrix(convoStorage.output, convoStorage.dilated4Filter, stride - 1,
                      std::make_pair(0, 0));

    // std::cout << "dilated out for input: " << std::endl;
    // std::cout << convoStorage.dilated4Input << std::endl;
    // std::cout << "dilated out for filter: " << std::endl;
    // std::cout << convoStorage.dilated4Filter << std::endl;

    FloatMatrix &mat_filter = filter.getMatrix();

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

    // compute input error
    rowsPos = 0;
    colsPos = 0;
    for (size_t i = 0; i < input.getRows(); i++) {
      for (size_t j = 0; j < input.getCols(); j++) {
        float sum = 0.f;
        for (long k = 0; k < filter.getRows(); k++) {
          for (long l = 0; l < filter.getCols(); l++) {
            sum += convoStorage.dilated4Input(k + rowsPos, l + colsPos) *
                   mat_filter(std::abs(k - (long) filter.getRows() + 1),
                              std::abs(l - (long) filter.getCols() + 1));
          }
        }
        storage.errorInput(i, j) = sum;
        colsPos++;
      }
      rowsPos++;
      colsPos = 0;
    }

    for (size_t i = 0; i < filter.getRows(); i++)
      for (size_t j = 0; j < filter.getCols(); j++)
        mat_filter(i, j) -= 0.2f * convoStorage.errorFilter(i, j);
  }


  CNNPoolingLayer::CNNPoolingLayer(const std::pair<size_t, size_t> poolSize, const size_t stride)
      : poolingSize(poolSize), CNNLayer(stride) {}


  CNNMaxPoolingLayer::CNNMaxPoolingLayer(const std::pair<size_t, size_t> poolSize,
                                         const size_t stride)
      : CNNPoolingLayer(poolSize, stride) {}

  void CNNMaxPoolingLayer::compute(const clFMatrix &_input, clFMatrix &_output) {
    FloatMatrix input = _input.toFloatMatrix(true);
    FloatMatrix output = _output.toFloatMatrix(true);

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

    _output = output;
  }

  void CNNMaxPoolingLayer::computeForward(const clFMatrix &_input, CNNStorageBP &storage) {
    FloatMatrix input = _input.toFloatMatrix(true);

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

  void CNNMaxPoolingLayer::computeBackward(const clFMatrix &_input, CNNStorageBP &storage) {
    FloatMatrix input = _input.toFloatMatrix(true);

    std::cout << "compute backprop max pooling" << std::endl;
    auto &poolingStorage = static_cast<CNNStorageBPMaxPooling &>(storage);

    for (auto &i : poolingStorage.errorInput) { i = 0.f; }

    for (size_t i = 0; i < poolingStorage.output.getRows(); i++) {
      for (size_t j = 0; j < poolingStorage.output.getCols(); j++) {
        poolingStorage.errorInput(poolingStorage.maxIndex(i, j).first,
                                  poolingStorage.maxIndex(i, j).second) +=
                poolingStorage.output(i, j);
      }
    }
  }

  CNNAvgPoolingLayer::CNNAvgPoolingLayer(const std::pair<size_t, size_t> poolSize,
                                         const size_t stride)
      : CNNPoolingLayer(poolSize, stride) {}

  void CNNAvgPoolingLayer::compute(const clFMatrix &_input, clFMatrix &_output) {
    FloatMatrix input = _input.toFloatMatrix(true);
    FloatMatrix output = _output.toFloatMatrix(true);

    size_t rowsPos = 0, colsPos = 0;

    for (size_t i = 0; i < output.getRows(); i++) {
      for (size_t j = 0; j < output.getCols(); j++) {
        float sum = 0.f;
        for (size_t k = 0; k < poolingSize.first; k++) {
          for (size_t l = 0; l < poolingSize.second; l++) {
            sum += input(k + rowsPos, l + colsPos);
          }
        }
        output(i, j) = sum / (float) (poolingSize.first * poolingSize.second);
        colsPos += stride;
      }
      colsPos = 0;
      rowsPos += stride;
    }

    _output = output;
  }

  void CNNAvgPoolingLayer::computeForward(const clFMatrix &input, CNNStorageBP &storage) {
    clFMatrix tmp = storage.output;
    compute(input, tmp);
    storage.output = tmp.toFloatMatrix(true);
  }

  void CNNAvgPoolingLayer::computeBackward(const clFMatrix &_input, CNNStorageBP &storage) {
    FloatMatrix input = _input.toFloatMatrix(true);

    auto &poolingStorage = static_cast<CNNStorageBPMaxPooling &>(storage);

    for (auto &i : poolingStorage.errorInput) { i = 0.f; }

    const float factor = 1.f / (float) (poolingSize.first * poolingSize.second);
    size_t rowsPos = 0, colsPos = 0;
    for (size_t i = 0; i < poolingStorage.output.getRows(); i++) {
      for (size_t j = 0; j < poolingStorage.output.getCols(); j++) {
        for (size_t k = 0; k < poolingSize.first; k++) {
          for (size_t l = 0; l < poolingSize.second; l++) {
            poolingStorage.errorInput(k + rowsPos, l + colsPos) +=
                    poolingStorage.output(i, j) * factor;
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

}   // namespace nnet