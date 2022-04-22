#include "CNN.hpp"

namespace nnet {

  std::vector<std::unique_ptr<CNNLayer>> CNN::copyLayers() {
    std::vector<std::unique_ptr<CNNLayer>> res;

    for (auto &layer : layers) { res.push_back(layer->copy()); }
    return res;
  }

  void CNN::setTopology(CNNTopology const &cnn_topology) {
    topology = cnn_topology;
    layers = topology.convertToLayer();
  }

  void CNN::randomizeWeight() {   // assert(false && "Not implemented");
    for (auto &layer : layers) {
      if (layer->hasWeight()) {
        math::FloatMatrix buffer(layer->getWeight().getRows(), layer->getWeight().getCols());
        for (size_t j = 0; j < layer->getWeight().getDepth(); j++) {
          math::randomize<float>(buffer, 0.f, 1.f);
          // TODO : check if we need to block operation
          layer->getWeight()[j].fromFloatMatrix(buffer, true);
        }
      }
    }
  }


  math::clFTensor CNN::predict(cl::CommandQueue &queue, math::clFTensor const &inputs) {
    if (layers.empty()) { throw std::runtime_error("CNN::predict : No layer in cnn"); }

    math::clFTensor output = inputs.shallowCopy();

    for (auto &layer : layers) {
      output = layer->compute(queue, output);
      //queue.finish();
      //std::cout << "output INTERMEDIAIRE : " << output << std::endl;
    }

    reorganizeForward(queue, output, inputs.getDepth(), topology.getNBranchFinal());

    queue.finish();

    return output;
  }

  void reorganizeForward(cl::CommandQueue &queue, math::clFTensor &tensor, const size_t nInput,
                         const size_t nBranchFinal) {
    if (nInput < 2) {
      tensor.reshape(tensor.getRows() * tensor.getCols() * (tensor.getDepth() / nInput), 1, nInput);
      return;
    }

    const size_t size_matrix = tensor.getRows() * tensor.getRows() * sizeof(float);
    math::clFTensor buffer(tensor.getRows(), tensor.getCols(), (nInput - 1) * nBranchFinal);

    size_t index = 0;
    for (size_t i = 1; i < nInput; i++) {
      for (size_t j = 0; j < nBranchFinal; j++) {
        queue.enqueueCopyBuffer(tensor.getBuffer(), buffer.getBuffer(),
                                tensor.getOffsetInBytes() + (i + j * nInput) * size_matrix,
                                buffer.getOffsetInBytes() + index * size_matrix, size_matrix);
        index++;
      }
    }

    for (size_t i = 1; i < nBranchFinal; i++) {
      queue.enqueueCopyBuffer(tensor.getBuffer(), tensor.getBuffer(),
                              tensor.getOffsetInBytes() + i * nInput * size_matrix,
                              tensor.getOffsetInBytes() + i * size_matrix, size_matrix);
    }

    queue.enqueueCopyBuffer(buffer.getBuffer(), tensor.getBuffer(), buffer.getOffsetInBytes(),
                            tensor.getOffsetInBytes() + nBranchFinal * size_matrix,
                            buffer.getDepth() * size_matrix);

    tensor.reshape(tensor.getRows() * tensor.getCols() * (tensor.getDepth() / nInput), 1, nInput);
  }

  void reorganizeBackward(cl::CommandQueue &queue, math::clFTensor &tensor, const size_t nInput,
                          const size_t nBranch, const std::pair<size_t, size_t> size) {
    tensor.reshape(size.first, size.second, nInput * nBranch);

    if (nInput < 2) return;

    const size_t size_matrix = tensor.getRows() * tensor.getRows() * sizeof(float);
    math::clFTensor buffer(tensor.getRows(), tensor.getCols(), (nInput - 1) * nBranch);

    queue.enqueueCopyBuffer(tensor.getBuffer(), buffer.getBuffer(),
                            tensor.getOffsetInBytes() + nBranch * size_matrix,
                            buffer.getOffsetInBytes(), buffer.getDepth() * size_matrix);


    for (size_t i = nBranch - 1; i > 0; i--) {
      queue.enqueueCopyBuffer(tensor.getBuffer(), tensor.getBuffer(),
                              tensor.getOffsetInBytes() + i * size_matrix,
                              tensor.getOffsetInBytes() + i * nInput * size_matrix, size_matrix);
    }

    size_t index = 0;
    for (size_t i = 1; i < nInput; i++) {
      for (size_t j = 0; j < nBranch; j++) {
        queue.enqueueCopyBuffer(buffer.getBuffer(), tensor.getBuffer(), index * size_matrix,
                                (i + j * nInput) * size_matrix, size_matrix);
        index++;
      }
    }
  }

}   // namespace nnet