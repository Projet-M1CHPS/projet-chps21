#include "CNNLayer.hpp"


namespace nnet {

  namespace {
    math::clFTensor reduceFilter(cl::CommandQueue &queue, const math::clFTensor &tensor,
                                 const size_t nInput, const size_t nFilter, const size_t nBranch) {
      const size_t n_total_filter = nFilter * nBranch;
      math::clFTensor res(tensor.getRows(), tensor.getCols(), n_total_filter);
      res.fill(0.f, queue, false);


      const size_t n = tensor.getRows() * tensor.getCols();
      std::vector<float> alphas(n_total_filter, 1.f);

      std::vector<size_t> x_offset(n_total_filter);
      std::vector<size_t> res_offset(n_total_filter);
      for (size_t i = 0; i < x_offset.size(); i++) {
        x_offset[i] = nInput * i * n;
        res_offset[i] = i * n;
      }

      for (size_t i = 0; i < nInput; i++) {
        clblast::AxpyBatched<float>(n, alphas.data(), tensor.getBuffer()(), x_offset.data(), 1,
                                    res.getBuffer()(), res_offset.data(), 1, n_total_filter,
                                    &queue(), nullptr);
        for (auto &val : x_offset) val += n;
      }
      res.ipscale(1.f / static_cast<float>(nInput), queue);
      return res;
    }

    math::clFTensor reduceInput(cl::CommandQueue &queue, const math::clFTensor &tensor,
                                const size_t nInput, const size_t nFilter, const size_t nBranch) {
      math::clFTensor res(tensor.getRows(), tensor.getCols(), nInput * nBranch);
      res.fill(0.f, queue, false);

      const size_t n = tensor.getRows() * tensor.getCols();
      std::vector<float> alphas(nInput, 1.f);

      std::vector<size_t> x_offset(nInput);
      std::vector<size_t> res_offset(nInput);
      for (size_t i = 0; i < x_offset.size(); i++) { x_offset[i] = res_offset[i] = i * n; }

      const size_t batch_count = nInput;

      for (size_t i = 0; i < nBranch; i++) {
        for (size_t j = 0; j < nFilter; j++) {
          clblast::AxpyBatched<float>(n, alphas.data(), tensor.getBuffer()(), x_offset.data(), 1,
                                      res.getBuffer()(), res_offset.data(), 1, batch_count,
                                      &queue(), nullptr);
          for (auto &val : x_offset) { val += batch_count * n; }
        }
        for (auto &val : res_offset) { val += batch_count * n; }
      }
      res.ipscale(1.f / static_cast<float>(nInput), queue);
      return res;
    }
  }   // namespace

  CNNLayer::CNNLayer(const std::pair<size_t, size_t> output) : outputSize(output) {}

  void CNNLayer::setWeight(const math::clFTensor &weights) {
    throw std::runtime_error("CNNLayer::setWeight: called on a filter without weights");
  }

  CNNConvolutionLayer::CNNConvolutionLayer(const std::pair<size_t, size_t> outputSize,
                                           const std::pair<size_t, size_t> sizeFilter,
                                           const size_t nFilter,
                                           const af::ActivationFunctionType aFunction,
                                           const size_t nBranch)
      : CNNLayer(outputSize), n_branch(nBranch), n_filter(nFilter),
        filters(sizeFilter.first, sizeFilter.second, nFilter * nBranch), a_function(aFunction) {}

  CNNConvolutionLayer::CNNConvolutionLayer(const CNNConvolutionLayer &other)
      : CNNLayer(other.outputSize), n_branch(other.n_branch), n_filter(other.n_filter),
        a_function(other.a_function) {
    // TODO : check if the copy is good
    filters.copy(other.filters, utils::cl_wrapper.getDefaultQueue(), true);
  }

  std::unique_ptr<CNNLayer> CNNConvolutionLayer::copy() const {
    return std::make_unique<CNNConvolutionLayer>(*this);
  }

  void CNNConvolutionLayer::setWeight(const math::clFTensor &weights) {
    filters.copy(weights, utils::cl_wrapper.getDefaultQueue(), true);
  }

  math::clFTensor CNNConvolutionLayer::compute(cl::CommandQueue &queue,
                                               const math::clFTensor &input) {
    std::cout << "call compute : branch{" << n_branch << "}, filter{" << n_filter << "}"
              << std::endl;

    const size_t output_size_z = n_branch * n_filter * input.getDepth() / n_branch;
    math::clFTensor res(outputSize.first, outputSize.second, output_size_z);

    auto sub_input = input.slice(n_branch);
    auto sub_filter = filters.slice(n_branch);
    auto sub_output = res.slice(n_branch * n_filter);

    size_t index_out = 0;
    for (size_t i = 0; i < n_branch; i++) {
      for (size_t j = 0; j < sub_filter[i].getDepth(); j++) {
        clblast::Convgemm<float>(clblast::KernelMode::kCrossCorrelation, 1, input.getRows(),
                                 input.getCols(), filters.getRows(), filters.getCols(), 0, 0, 1, 1,
                                 1, 1, 1, sub_input[i].getDepth(), sub_input[i].getBuffer()(),
                                 sub_input[i].getOffsetInFloats(), sub_filter[i][j].getBuffer()(),
                                 sub_filter[i][j].getOffset(), sub_output[index_out].getBuffer()(),
                                 sub_output[index_out].getOffsetInFloats(), &queue(), nullptr);
        index_out++;
      }
    }

    applyAF(a_function, res, queue);
    return res;
  }


  math::clFTensor CNNConvolutionLayer::computeForward(cl::CommandQueue &queue,
                                                      const math::clFTensor &inputs,
                                                      CNNStorageBP &storage) {
    auto &convoStorage = static_cast<CNNStorageBPConvolution &>(storage);
    convoStorage.input = inputs.shallowCopy();
    return compute(queue, inputs);
  }


  math::clFTensor CNNConvolutionLayer::computeBackward(cl::CommandQueue &queue,
                                                       const math::clFTensor &errors,
                                                       CNNStorageBP &storage) {
    auto &convoStorage = static_cast<CNNStorageBPConvolution &>(storage);
    const size_t n_input = convoStorage.input.getDepth() / n_branch;

    math::clFTensor res_filter = computeErrorFilter(queue, errors, convoStorage);

    math::clFTensor res_input = computeErrorInput(queue, errors, convoStorage);
    convoStorage.error_filter = reduceFilter(queue, res_filter, n_input, n_filter, n_branch);
    return reduceInput(queue, res_input, n_input, n_filter, n_branch);
  }

  math::clFTensor CNNConvolutionLayer::computeErrorFilter(cl::CommandQueue &queue,
                                                          const math::clFTensor &errors,
                                                          CNNStorageBPConvolution &storage) {
    // compute filter error
    math::clFTensor res_filter(filters.getRows(), filters.getCols(), errors.getDepth());
    const size_t height = storage.input.getRows();
    const size_t width = storage.input.getCols();
    const size_t kernel_h = errors.getRows();
    const size_t kernel_w = errors.getCols();
    const size_t n_input = storage.input.getDepth() / n_branch;

    std::vector<math::clFTensor> sub_input = storage.input.slice(n_branch);
    std::vector<math::clFTensor> sub_error = errors.slice(n_branch * n_filter);

    size_t index = 0;
    for (size_t i = 0; i < n_branch; i++) {
      for (size_t j = 0; j < n_filter; j++) {
        for (size_t k = 0; k < n_input; k++) {
          clblast::Convgemm<float>(clblast::KernelMode::kCrossCorrelation, 1, height, width,
                                   kernel_h, kernel_w, 0, 0, 1, 1, 1, 1, 1, 1,
                                   sub_input[i][k].getBuffer()(), sub_input[i][k].getOffset(),
                                   errors[index].getBuffer()(), errors[index].getOffset(),
                                   res_filter[index].getBuffer()(), res_filter[index].getOffset(),
                                   &queue(), nullptr);
          index++;
        }
      }
    }
    return res_filter.shallowCopy();
  }


  math::clFTensor CNNConvolutionLayer::computeErrorInput(cl::CommandQueue &queue,
                                                         const math::clFTensor &errors,
                                                         CNNStorageBPConvolution &storage) {
    math::clFTensor res_input(storage.input.getRows(), storage.input.getCols(), errors.getDepth());
    // compute input error
    const size_t height = errors.getRows();
    const size_t width = errors.getCols();
    const size_t kernel_h = filters.getRows();
    const size_t kernel_w = filters.getCols();
    const size_t batch_count = errors.getDepth() / (n_branch * n_filter);

    std::vector<math::clFTensor> sub_error = errors.slice(n_branch * n_filter);
    std::vector<math::clFTensor> sub_filter = filters.slice(n_branch * n_filter);

    size_t index = 0;
    for (size_t i = 0; i < n_branch; i++) {
      for (size_t j = 0; j < n_filter; j++) {
        clblast::Convgemm<float>(clblast::KernelMode::kConvolution, 1, height, width, kernel_h,
                                 kernel_w, kernel_h - 1, kernel_w - 1, 1, 1, 1, 1, 1, batch_count,
                                 sub_error[i * n_filter + j].getBuffer()(),
                                 sub_error[i * n_filter + j].getOffsetInFloats(),
                                 sub_filter[i * n_filter + j].getBuffer()(),
                                 sub_filter[i * n_filter + j].getOffsetInFloats(),
                                 res_input[index].getBuffer()(), res_input[index].getOffset(),
                                 &queue(), nullptr);
        index += batch_count;
      }
    }
    return res_input;
  }


  CNNPoolingLayer::CNNPoolingLayer(const std::pair<size_t, size_t> outputSize,
                                   const std::pair<size_t, size_t> poolSize)
      : CNNLayer(outputSize), poolingSize(poolSize) {}


  CNNMaxPoolingLayer::CNNMaxPoolingLayer(const std::pair<size_t, size_t> outputSize,
                                         const std::pair<size_t, size_t> poolSize)
      : CNNPoolingLayer(outputSize, poolSize) {}

  CNNMaxPoolingLayer::CNNMaxPoolingLayer(const CNNMaxPoolingLayer &other)
      : CNNPoolingLayer(other.outputSize, other.poolingSize) {}

  std::unique_ptr<CNNLayer> CNNMaxPoolingLayer::copy() const {
    return std::make_unique<CNNMaxPoolingLayer>(*this);
  }

  math::clFTensor CNNMaxPoolingLayer::compute(cl::CommandQueue &queue,
                                              const math::clFTensor &inputs) {
    math::clFTensor res(outputSize.first, outputSize.second, inputs.getDepth());
    for (size_t ii = 0; ii < inputs.getDepth(); ii++) {
      math::FloatMatrix _input = inputs[ii].toFloatMatrix(true);
      math::FloatMatrix _output = res[ii].toFloatMatrix(true);

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
          colsPos++;
        }
        colsPos = 0;
        rowsPos++;
      }
      res[ii] = _output;
    }
    return res;
  }

  math::clFTensor CNNMaxPoolingLayer::computeForward(cl::CommandQueue &queue,
                                                     const math::clFTensor &inputs,
                                                     CNNStorageBP &storage) {
    auto &poolingStorage = static_cast<CNNStorageBPMaxPooling &>(storage);
    math::clFTensor res(outputSize.first, outputSize.second, inputs.getDepth());
    for (size_t ii = 0; ii < inputs.getDepth(); ii++) {
      math::FloatMatrix _input = inputs[ii].toFloatMatrix(true);
      math::FloatMatrix _output = res[ii].toFloatMatrix(true);

      math::Matrix<size_t> save_rows(outputSize.first, outputSize.second);
      math::Matrix<size_t> save_cols(outputSize.first, outputSize.second);
      save_rows.fill(0);
      save_cols.fill(0);

      size_t rowsPos = 0, colsPos = 0;
      for (size_t i = 0; i < _output.getRows(); i++) {
        for (size_t j = 0; j < _output.getCols(); j++) {
          float max = _input(rowsPos, colsPos);
          save_rows(i, j) = rowsPos;
          save_cols(i, j) = colsPos;
          for (size_t k = 0; k < poolingSize.first; k++) {
            for (size_t l = 0; l < poolingSize.second; l++) {
              if (max < _input(k + rowsPos, l + colsPos)) {
                max = _input(k + rowsPos, l + colsPos);
                save_rows(i, j) = k + rowsPos;
                save_cols(i, j) = l + colsPos;
              }
            }
          }
          _output(i, j) = max;
          colsPos++;
        }
        colsPos = 0;
        rowsPos++;
      }
      poolingStorage.max_rows.push_back(std::move(save_rows));
      poolingStorage.max_cols.push_back(std::move(save_cols));
      res[ii] = _output;
    }
    return res;
  }

  math::clFTensor CNNMaxPoolingLayer::computeBackward(cl::CommandQueue &queue,
                                                      const math::clFTensor &errors,
                                                      CNNStorageBP &storage) {
    auto &poolingStorage = static_cast<CNNStorageBPMaxPooling &>(storage);

    math::clFTensor res(poolingStorage.input_size.first, poolingStorage.input_size.second,
                        errors.getDepth());
    for (size_t ii = 0; ii < errors.getDepth(); ii++) {
      math::FloatMatrix error_output = errors[ii].toFloatMatrix(true);
      math::FloatMatrix error_input = res[ii].toFloatMatrix(true);
      error_input.fill(0.f);

      math::Matrix<size_t> &save_rows = poolingStorage.max_rows[ii];
      math::Matrix<size_t> &save_cols = poolingStorage.max_cols[ii];

      for (size_t i = 0; i < errors.getRows(); i++) {
        for (size_t j = 0; j < errors.getCols(); j++) {
          error_input(save_rows(i, j), save_cols(i, j)) += error_output(i, j);
        }
      }
      res[ii] = error_input;
    }
    return res;
  }

  CNNAvgPoolingLayer::CNNAvgPoolingLayer(const std::pair<size_t, size_t> outputSize,
                                         const std::pair<size_t, size_t> poolSize)
      : CNNPoolingLayer(outputSize, poolSize), filter(poolSize.first, poolSize.second, 1) {
    // TODO : faire attention a la queu sur laquel on fait le calcul
    filter[0].fill(1.f, utils::cl_wrapper.getDefaultQueue(), true);
  }

  CNNAvgPoolingLayer::CNNAvgPoolingLayer(const CNNAvgPoolingLayer &other)
      : CNNPoolingLayer(other.outputSize, other.poolingSize) {
    // TODO : je crois je fais de ma merde
    filter.copy(other.filter, utils::cl_wrapper.getDefaultQueue(), true);
  }

  std::unique_ptr<CNNLayer> CNNAvgPoolingLayer::copy() const {
    return std::make_unique<CNNAvgPoolingLayer>(*this);
  }

  math::clFTensor CNNAvgPoolingLayer::compute(cl::CommandQueue &queue,
                                              const math::clFTensor &input) {
    math::clFTensor res(outputSize.first, outputSize.second, input.getDepth());

    clblast::Convgemm<float>(clblast::KernelMode::kCrossCorrelation, 1, input.getRows(),
                             input.getCols(), filter.getRows(), filter.getCols(), 0, 0, 1, 1, 1, 1,
                             1, input.getDepth(), input.getBuffer()(), input.getOffsetInFloats(),
                             filter.getBuffer()(), filter.getOffsetInFloats(), res.getBuffer()(),
                             res.getOffsetInFloats(), &queue(), nullptr);

    const float scale = 1.f / static_cast<float>(filter.getRows() * filter.getCols());
    res.ipscale(scale, queue);

    return res;
  }

  math::clFTensor CNNAvgPoolingLayer::computeForward(cl::CommandQueue &queue,
                                                     const math::clFTensor &inputs,
                                                     CNNStorageBP &storages) {
    return compute(queue, inputs);
  }

  math::clFTensor CNNAvgPoolingLayer::computeBackward(cl::CommandQueue &queue,
                                                      const math::clFTensor &errors,
                                                      CNNStorageBP &storages) {
    auto &poolingStorage = static_cast<CNNStorageBPAvgPooling &>(storages);

    math::clFTensor res(poolingStorage.input_size.first, poolingStorage.input_size.second,
                        errors.getDepth());

    for (size_t ii = 0; ii < errors.getDepth(); ii++) {
      math::FloatMatrix error_output = errors[ii].toFloatMatrix(true);
      math::FloatMatrix error_input = res[ii].toFloatMatrix(true);
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
          colsPos++;
        }
        colsPos = 0;
        rowsPos++;
      }
      error_input *= 1.f / (float) (poolingSize.first * poolingSize.second);
      res[ii] = error_input;
    }
    return res;
  }


}   // namespace nnet