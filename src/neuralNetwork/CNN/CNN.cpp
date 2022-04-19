#include "CNN.hpp"

namespace nnet {

  std::vector<std::unique_ptr<CNNLayer>> CNN::copyLayers() {
    std::vector<std::unique_ptr<CNNLayer>> res;

    for (auto &layer : layers) { res.push_back(layer->copy()); }
    return res;
  }

  std::vector<clFTensor> CNN::getWeights() {
    std::vector<clFTensor> res;
    for (auto &layer : layers) { layer->getWeight(res); }
    return res;
  }

  void CNN::setWeights(std::vector<clFTensor> &tensors) {
    size_t index = 0;
    for (auto &layer : layers) {
      if (layer->setWeight(tensors[index])) index++;
    }
  }

  void CNN::setTopology(CNNTopology const &cnn_topology) {
    topology = cnn_topology;
    layers = topology.convertToLayer();
  }

  void CNN::randomizeWeight() { assert(false && "Not implemented"); }


  clFTensor CNN::predict(clFTensor const &inputs) {
    /*if (layers.empty()) { throw std::runtime_error("no layer in cnn"); }

    clFTensor output = inputs.shallowCopy();

    for (auto &layer : layers) {
      output = layer->compute(output);
      std::cout << "output INTERMEDIAIRE : " << output << std::endl;
    }

    reorganizeForward(output, inputs.getDepth(), topology.getBranchFinal());

    utils::cl_wrapper.getDefaultQueue().finish();

    return output;*/

    auto queue = utils::cl_wrapper.getDefaultQueue();
    clFTensor tensor(3, 3, 12);

    tensor[0].fill(1.1f, queue);
    tensor[1].fill(2.1f, queue);
    tensor[2].fill(3.1f, queue);
    tensor[3].fill(1.2f, queue);
    tensor[4].fill(2.2f, queue);
    tensor[5].fill(3.2f, queue);
    tensor[6].fill(1.3f, queue);
    tensor[7].fill(2.3f, queue);
    tensor[8].fill(3.3f, queue);
    tensor[9].fill(1.4f, queue);
    tensor[10].fill(2.4f, queue);
    tensor[11].fill(3.4f, queue);

    queue.finish();
    std::cout << "before : " << tensor << std::endl;

    reorganizeForward(queue, tensor, 3, 4);

    queue.finish();
    std::cout << "after : " << tensor << std::endl;

    reorganizeBackward(queue, tensor, 3, 4, {3, 3});

    queue.finish();
    std::cout << "after after : " << tensor << std::endl;

    exit(12);
  }

  void reorganizeForward(cl::CommandQueue &queue, clFTensor &tensor, const size_t nInput,
                         const size_t nBranch) {
    if (nInput < 2 || nBranch < 2) return;

    const size_t size_matrix = tensor.getRows() * tensor.getRows() * sizeof(float);
    clFTensor buffer(tensor.getRows(), tensor.getCols(), (nInput - 1) * nBranch);

    size_t index = 0;
    for (size_t i = 1; i < nInput; i++) {
      for (size_t j = 0; j < nBranch; j++) {
        queue.enqueueCopyBuffer(tensor.getBuffer(), buffer.getBuffer(),
                                tensor.getOffsetInBytes() + (i + j * nInput) * size_matrix,
                                buffer.getOffsetInBytes() + index * size_matrix, size_matrix);
        index++;
      }
    }

    for (size_t i = 1; i < nBranch; i++) {
      queue.enqueueCopyBuffer(tensor.getBuffer(), tensor.getBuffer(),
                              tensor.getOffsetInBytes() + i * nInput * size_matrix,
                              tensor.getOffsetInBytes() + i * size_matrix, size_matrix);
    }

    queue.enqueueCopyBuffer(buffer.getBuffer(), tensor.getBuffer(), buffer.getOffsetInBytes(),
                            tensor.getOffsetInBytes() + nBranch * size_matrix,
                            buffer.getDepth() * size_matrix);

    tensor.reshape(nBranch * tensor.getRows() * tensor.getCols(), 1, nInput);
  }

  void reorganizeBackward(cl::CommandQueue &queue, clFTensor &tensor, const size_t nInput,
                          const size_t nBranch, const std::pair<size_t, size_t> size) {
    if (nInput < 2 || nBranch < 2) return;

    tensor.reshape(size.first, size.second, nInput * nBranch);

    const size_t size_matrix = tensor.getRows() * tensor.getRows() * sizeof(float);
    clFTensor buffer(tensor.getRows(), tensor.getCols(), (nInput - 1) * nBranch);

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