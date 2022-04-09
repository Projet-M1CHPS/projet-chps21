#pragma once
#include "clUtils/clFTensor.hpp"
#include <iostream>
#include <vector>

namespace nnet {

  class BatchScheduler {
  public:
    BatchScheduler(size_t batch_size) : batch_size(batch_size) {}

    void run(const std::vector<math::clFTensor> &inputs,
             const std::vector<math::clFTensor> &targets) {
      // TODO: implement this
      // Test function using stochastic gradient descent

      auto &queue = utils::cl_wrapper.getDefaultQueue();
      for (size_t i = 0; i < inputs.size(); i++) {
        runBatch(inputs[i], targets[i], queue);
        endOfBatch();
      }
    }

    virtual void endOfBatch() = 0;

    virtual void runBatch(const math::clFTensor &inputs, const math::clFTensor &targets,
                          cl::CommandQueue &queue) = 0;

  protected:
    size_t batch_size;
  };

}   // namespace nnet
