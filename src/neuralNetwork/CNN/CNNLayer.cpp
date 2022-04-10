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
        n_filter(nFilter), padding(padding), activationFunction(aFunction) {}


  clFTensor CNNConvolutionLayer::compute(const clFTensor &input) {
    auto sub_tensor_in = input.shallowSplit(n_branch);

    std::cout << "call compute : branch{" << n_branch << "}, filter{" << n_filter << "}"
              << std::endl;

    cl::CommandQueue queue = utils::cl_wrapper.getDefaultQueue();

    const size_t output_size_z = n_branch * n_filter * input.getZ() / n_branch;
    clFTensor res(outputSize.first, outputSize.second, output_size_z);

    auto sub_filter = filters.shallowSplit(n_branch);
    auto sub_tensor_out = res.shallowSplit(n_branch * n_filter);

    size_t out_index = 0;
    for (size_t i = 0; i < n_branch; i++) {
      std::cout << "branch" << std::endl;
      for (size_t j = 0; j < n_filter; j++) {
        std::cout << "filter" << std::endl;

        std::cout << sub_filter[i].getMatrix(0).toFloatMatrix(true) << std::endl;

        clblast::Convgemm<float>(
                clblast::KernelMode::kCrossCorrelation, 1, input.getX(), input.getY(),
                filters.getX(), filters.getY(), padding, padding, stride, stride, 1, 1, 1,
                input.getZ() / n_branch, sub_tensor_in[i].getBuffer()(),
                sub_tensor_in[i].offset * input.getX() * input.getY(),
                sub_filter[i].getMatrix(j).getBuffer()(), sub_filter[i].getMatrix(j).getOffset(),
                sub_tensor_out[out_index].getBuffer()(),
                sub_tensor_out[out_index].offset * res.getX() * res.getY(), &queue(), nullptr);
        out_index++;
      }
    }

    // applyAF(activationFunction, res, queue);
    return res;
  }


  clFTensor CNNConvolutionLayer::computeForward(const clFTensor &input, CNNStorageBP &storage) {
    return compute(input);
  }


  clFTensor CNNConvolutionLayer::computeBackward(const clFTensor &errors, CNNStorageBP &storage) {
    auto &convoStorage = static_cast<CNNStorageBPConvolution &>(storage);

    clFTensor res(3, 3, errors.getZ());

    cl::CommandQueue queue = utils::cl_wrapper.getDefaultQueue();

    {
      const size_t height = convoStorage.input.getX();
      const size_t width = convoStorage.input.getY();
      const size_t kernel_h = errors.getX();
      const size_t kernel_w = errors.getY();
      const size_t dilatation_h = stride;
      const size_t dilatation_w = stride;

      // compute filter error
      clblast::Convgemm<float>(clblast::KernelMode::kCrossCorrelation, 1, height, width, kernel_h,
                               kernel_w, 0, 0, 1, 1, dilatation_h, dilatation_w, 1, 1,
                               convoStorage.input.getBuffer()(), 0, errors.getBuffer()(), 0,
                               convoStorage.errorFilter.getBuffer()(), 0, &queue(), nullptr);
    }

    // compute input error
    // on convertie le tensor en matrice dilater



    /*rowsPos = 0;
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
    }*/


    return res;
  }


  CNNPoolingLayer::CNNPoolingLayer(const std::pair<size_t, size_t> outputSize,
                                   const std::pair<size_t, size_t> poolSize, const size_t stride)
      : poolingSize(poolSize), CNNLayer(outputSize, stride) {}


  CNNMaxPoolingLayer::CNNMaxPoolingLayer(const std::pair<size_t, size_t> outputSize,
                                         const std::pair<size_t, size_t> poolSize,
                                         const size_t stride)
      : CNNPoolingLayer(outputSize, poolSize, stride) {}

  clFTensor CNNMaxPoolingLayer::compute(const clFTensor &inputs) {
    clFTensor res(outputSize.first, outputSize.second, inputs.getZ());
    for (size_t ii = 0; ii < inputs.getZ(); ii++) {
      FloatMatrix _input = inputs.getMatrix(ii).toFloatMatrix(true);
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
      res.getMatrix(ii) = _output;
    }
    return res;
  }

  clFTensor CNNMaxPoolingLayer::computeForward(const clFTensor &inputs, CNNStorageBP &storage) {
    auto &poolingStorage = static_cast<CNNStorageBPMaxPooling &>(storage);
    clFTensor res(outputSize.first, outputSize.second, inputs.getZ());
    for (size_t ii = 0; ii < inputs.getZ(); ii++) {
      FloatMatrix _input = inputs.getMatrix(ii).toFloatMatrix(true);
      FloatMatrix _output = res.getMatrix(ii).toFloatMatrix(true);

      Matrix<size_t> save_rows(outputSize.first, outputSize.second);
      Matrix<size_t> save_cols(outputSize.first, outputSize.second);
      save_rows.fill(0);
      save_cols.fill(0);

      size_t rowsPos = 0, colsPos = 0;
      std::cout << "put" << _input << std::endl;
      for (size_t i = 0; i < _output.getRows(); i++) {
        for (size_t j = 0; j < _output.getCols(); j++) {
          float max = _input(rowsPos, colsPos);
          save_rows(i, j) = rowsPos;
          save_cols(i, j) = colsPos;
          for (size_t k = 0; k < poolingSize.first; k++) {
            for (size_t l = 0; l < poolingSize.second; l++) {
              std::cout << max << " " << _input(k + rowsPos, l + colsPos) << " " << k + rowsPos
                        << " " << l + colsPos << std::endl;
              if (max < _input(k + rowsPos, l + colsPos)) {
                max = _input(k + rowsPos, l + colsPos);
                save_rows(i, j) = k + rowsPos;
                save_cols(i, j) = l + colsPos;
              }
            }
          }
          _output(i, j) = max;
          colsPos += stride;
        }
        colsPos = 0;
        rowsPos += stride;
      }
      poolingStorage.max_rows.push_back(std::move(save_rows));
      poolingStorage.max_cols.push_back(std::move(save_cols));
      res.getMatrix(ii) = _output;
    }
    return res;
  }

  clFTensor CNNMaxPoolingLayer::computeBackward(const clFTensor &errors, CNNStorageBP &storage) {
    auto &poolingStorage = static_cast<CNNStorageBPMaxPooling &>(storage);

    clFTensor res(poolingStorage.input_size.first, poolingStorage.input_size.second, errors.getZ());
    for (size_t ii = 0; ii < errors.getZ(); ii++) {
      FloatMatrix error_output = errors.getMatrix(ii).toFloatMatrix(true);
      FloatMatrix error_input = res.getMatrix(ii).toFloatMatrix(true);
      error_input.fill(0.f);

      Matrix<size_t> &save_rows = poolingStorage.max_rows[ii];
      Matrix<size_t> &save_cols = poolingStorage.max_cols[ii];

      for (size_t i = 0; i < errors.getX(); i++) {
        for (size_t j = 0; j < errors.getY(); j++) {
          error_input(save_rows(i, j), save_cols(i, j)) += error_output(i, j);
        }
      }
      res.getMatrix(ii) = error_input;
    }
    return res;
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

  clFTensor CNNAvgPoolingLayer::computeForward(const clFTensor &input, CNNStorageBP &storages) {
    return compute(input);
  }

  clFTensor CNNAvgPoolingLayer::computeBackward(const clFTensor &errors, CNNStorageBP &storages) {
    auto &poolingStorage = static_cast<CNNStorageBPAvgPooling &>(storages);

    clFTensor res(poolingStorage.input_size.first, poolingStorage.input_size.second, errors.getZ());

    for (size_t ii = 0; ii < errors.getZ(); ii++) {
      std::cout << "call" << std::endl;
      FloatMatrix error_output = errors.getMatrix(ii).toFloatMatrix(true);
      FloatMatrix error_input = res.getMatrix(ii).toFloatMatrix(true);
      error_input.fill(0.f);

      size_t rowsPos = 0, colsPos = 0;

      const size_t max_i = error_input.getRows() - error_output.getRows() + 1;
      const size_t max_j = error_input.getCols() - error_output.getCols() + 1;
      for (size_t i = 0; i < max_i; i++) {
        for (size_t j = 0; j < max_j; j++) {
          for (size_t k = 0; k < error_output.getRows(); k++) {
            for (size_t l = 0; l < error_output.getCols(); l++) {
              error_input(k + rowsPos, l + colsPos) += error_output(i, j);
            }
          }
          colsPos += stride;
        }
        colsPos = 0;
        rowsPos += stride;
      }
      error_input *= 1.f / (float) (poolingSize.first * poolingSize.second);
      res.getMatrix(ii) = error_input;
    }
    return res;
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