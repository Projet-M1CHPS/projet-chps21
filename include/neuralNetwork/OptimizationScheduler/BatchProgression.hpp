#pragma once
#include "math/clFTensor.hpp"
#include <vector>

namespace nnet {

  class BatchProgression {
  public:
    BatchProgression(const std::vector<math::clFTensor> &inputs,
                     const std::vector<math::clFTensor> &targets, size_t tensor_index = 0,
                     size_t local_index = 0);

    void progress(size_t count);

    const std::vector<math::clFTensor> &getInputs() const { return *inputs; }

    const std::vector<math::clFTensor> &getTargets() const { return *targets; }

    const math::clFTensor &getInput(size_t index) const { return (*inputs)[index]; }

    size_t getBatchRemainder() const { return (*inputs)[tensor_index].getDepth() - local_index; }

    math::clFTensor getInputSlice(size_t size) const {
      return (*inputs)[tensor_index].slice(local_index, local_index + size);
    }

    const math::clFTensor &getTarget(size_t index) const { return (*targets)[index]; }

    math::clFTensor getTargetSlice(size_t size) const {
      return (*targets)[tensor_index].slice(local_index, local_index + size);
    }

    size_t getTensorIndex() const { return tensor_index; }

    size_t getLocalIndex() const { return local_index; }

  private:
    const std::vector<math::clFTensor> *inputs, *targets;
    size_t tensor_index, local_index;
  };

}   // namespace nnet
