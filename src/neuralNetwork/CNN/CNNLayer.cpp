#include "CNNLayer.hpp"


namespace nnet {

  CNNLayer::CNNLayer(const std::pair<size_t, size_t> output, const size_t stride)
      : outputSize(output), stride(stride) {}


  CNNConvolutionLayer::CNNConvolutionLayer(const std::pair<size_t, size_t> outputSize,
                                           const std::pair<size_t, size_t> sizeFilter,
                                           const size_t nFilter,
                                           const af::ActivationFunctionType aFunction,
                                           const size_t stride, const size_t nBranch,
                                           const size_t padding)
      : CNNLayer(outputSize, stride),
        filters(sizeFilter.first, sizeFilter.second, nFilter * nBranch), n_branch(nBranch),
        padding(padding), activationFunction(aFunction) {
  }


  clFTensor CNNConvolutionLayer::compute(const clFTensor &input) {
    auto sub_tensor_in = input.shallowSplit(n_branch);

    clFTensor res(outputSize.first, outputSize.second, n_branch * filters.getZ());
    auto sub_tensor_out = res.shallowSplit(n_branch * filters.getZ());

    cl::CommandQueue queue = utils::cl_wrapper.getDefaultQueue();
    size_t out_index = 0;
    for (size_t i = 0; i < n_branch; i++) {
      for (size_t j = 0; j < filters.getZ(); j++) {
        clblast::Convgemm<float>(
                clblast::KernelMode::kCrossCorrelation, 1, input.getX(), input.getY(),
                filters.getX(), filters.getY(), padding, padding, stride, stride, 1, 1, 1,
                input.getZ(), sub_tensor_in[i].getBuffer()(), 0, filters.getMatrix(j).getBuffer()(),
                0, sub_tensor_out[out_index++].getBuffer()(), 0, &queue(), nullptr);
      }
    }

    // applyAF(activationFunction, res, queue);
    return res;
  }


  clFTensor CNNConvolutionLayer::computeForward(const clFTensor &input, CNNStorageBP &storage) {
    return compute(input);
  }


  clFTensor CNNConvolutionLayer::computeBackward(const clFTensor &input, CNNStorageBP &storage) {
    /*FloatMatrix input = _input.toFloatMatrix(true);

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
        mat_filter(i, j) -= 0.2f * convoStorage.errorFilter(i, j);*/

    return {0, 0, 0};
  }


  CNNPoolingLayer::CNNPoolingLayer(const std::pair<size_t, size_t> outputSize,
                                   const std::pair<size_t, size_t> poolSize, const size_t stride)
      : poolingSize(poolSize), CNNLayer(outputSize, stride) {}


  CNNMaxPoolingLayer::CNNMaxPoolingLayer(const std::pair<size_t, size_t> outputSize,
                                         const std::pair<size_t, size_t> poolSize,
                                         const size_t stride)
      : CNNPoolingLayer(outputSize, poolSize, stride) {}

  clFTensor CNNMaxPoolingLayer::compute(const clFTensor &input) {
    clFTensor res(outputSize.first, outputSize.second, input.getZ());
    for (size_t ii = 0; ii < input.getZ(); ii++) {
      FloatMatrix _input = input.getMatrix(ii).toFloatMatrix(true);
      FloatMatrix _output = res.getMatrix(ii).toFloatMatrix(true);

      size_t rowsPos = 0, colsPos = 0;

      for (size_t i = 0; i < _output.getRows(); i++) {
        for (size_t j = 0; j < _output.getCols(); j++) {
          float max = _input(rowsPos, colsPos);
          for (size_t k = 0; k < poolingSize.first; k++) {
            for (size_t l = 0; l < poolingSize.second; l++) {
              max = std::max(max, _input(k + rowsPos, l + colsPos));
            }
          }
          _output(i, j) = max;
          colsPos += stride;
        }
        colsPos = 0;
        rowsPos += stride;
      }
      // TODO : faire attention je crois on modifie pas le tensor res
      res.getMatrix(ii) = _output;
    }
    return res;
  }

  clFTensor CNNMaxPoolingLayer::computeForward(const clFTensor &input, CNNStorageBP &storage) {
    /*FloatMatrix input = _input.toFloatMatrix(true);

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
    }*/
    return {0, 0, 0};
  }

  clFTensor CNNMaxPoolingLayer::computeBackward(const clFTensor &_input, CNNStorageBP &storage) {
    /*FloatMatrix input = _input.toFloatMatrix(true);

    std::cout << "compute backprop max pooling" << std::endl;
    auto &poolingStorage = static_cast<CNNStorageBPMaxPooling &>(storage);

    for (auto &i : poolingStorage.errorInput) { i = 0.f; }

    for (size_t i = 0; i < poolingStorage.output.getRows(); i++) {
      for (size_t j = 0; j < poolingStorage.output.getCols(); j++) {
        poolingStorage.errorInput(poolingStorage.maxIndex(i, j).first,
                                  poolingStorage.maxIndex(i, j).second) +=
                poolingStorage.output(i, j);
      }
    }*/
    return {0, 0, 0};
  }

  CNNAvgPoolingLayer::CNNAvgPoolingLayer(const std::pair<size_t, size_t> outputSize,
                                         const std::pair<size_t, size_t> poolSize,
                                         const size_t stride)
      : CNNPoolingLayer(outputSize, poolSize, stride) {}

  clFTensor CNNAvgPoolingLayer::compute(const clFTensor &input) {
    clFTensor res(outputSize.first, outputSize.second, input.getZ());
    for (size_t ii = 0; ii < input.getZ(); ii++) {
      FloatMatrix _input = input.getMatrix(ii).toFloatMatrix(true);
      FloatMatrix _output = res.getMatrix(ii).toFloatMatrix(true);

      size_t rowsPos = 0, colsPos = 0;

      for (size_t i = 0; i < outputSize.first; i++) {
        for (size_t j = 0; j < outputSize.second; j++) {
          float sum = 0.f;
          for (size_t k = 0; k < poolingSize.first; k++) {
            for (size_t l = 0; l < poolingSize.second; l++) {
              sum += _input(k + rowsPos, l + colsPos);
            }
          }
          _output(i, j) = sum / static_cast<float>(poolingSize.first * poolingSize.second);
          colsPos += stride;
        }
        colsPos = 0;
        rowsPos += stride;
      }
      res.getMatrix(ii) = _output;
    }
    return res;
  }

  clFTensor CNNAvgPoolingLayer::computeForward(const clFTensor &input, CNNStorageBP &storage) {
    return compute(input);
  }

  clFTensor CNNAvgPoolingLayer::computeBackward(const clFTensor &_input, CNNStorageBP &storage) {
    /*FloatMatrix input = _input.toFloatMatrix(true);

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
    }*/
    return {0, 0, 0};
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