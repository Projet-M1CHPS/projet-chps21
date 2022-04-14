#include "CNN.hpp"

namespace nnet {

  namespace {
    void reorganizeForward(clFTensor &tensor, const size_t nInput, const size_t nBranch) {
      if (nInput < 2 || nBranch < 2) return;

      const size_t size = tensor.getRows() * tensor.getRows();
      clFTensor buffer(tensor.getRows(), tensor.getCols(), (nInput - 1) * nBranch);
      size_t index = 0;

      for (size_t i = 1; i < nInput; i++) {
        for (size_t j = 0; j < nBranch; j++) {
          cl::enqueueCopyBuffer(tensor.getBuffer(), buffer.getBuffer(),
                                ((i + j * nInput) * size) * sizeof(float),
                                (index * size) * sizeof(float), size * sizeof(float));
          index++;
        }
      }

      std::cout << "buffer : " << buffer << std::endl;

      for (size_t i = 1; i < nBranch; i++) {
        cl::enqueueCopyBuffer(tensor.getBuffer(), tensor.getBuffer(),
                              tensor.getOffsetInBytes() + i * nInput * size * sizeof(float),
                              tensor.getOffsetInBytes() + i * size * sizeof(float),
                              size * sizeof(float));
      }
      auto queue = utils::cl_wrapper.getDefaultQueue();
      cl::enqueueCopyBuffer(buffer.getBuffer(), tensor.getBuffer(), buffer.getOffsetInBytes(),
                            tensor.getOffsetInBytes() + nBranch * size * sizeof(float),
                            buffer.getDepth() * size * sizeof(float));

      tensor.reshape(nBranch * tensor.getRows() * tensor.getCols(), 1, nInput);
    }

  }   // namespace

  void CNN::setTopology(CNNTopology const &cnn_topology) {
    topology = cnn_topology;
    layers = topology.convertToLayer();
  }


  void CNN::randomizeWeight() { assert(false && "Not implemented"); }


  clFTensor CNN::predict(clFTensor const &inputs) {
    if (layers.empty()) { throw std::runtime_error("no layer in cnn"); }

    clFTensor output = inputs.shallowCopy();

    for (auto &layer : layers) {
      output = layer->compute(output);
      std::cout << "output INTERMEDIAIRE : " << output << std::endl;
    }

    reorganizeForward(output, inputs.getDepth(), topology.getBranchFinal());

    utils::cl_wrapper.getDefaultQueue().finish();

    return output;

    /*auto queue = utils::cl_wrapper.getDefaultQueue();
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

    std::cout << "before : " << tensor << std::endl;

    reorganizeForward(tensor, 3, 4);

    std::cout << "after : " << tensor << std::endl;

    exit(12);*/
  }

}   // namespace nnet