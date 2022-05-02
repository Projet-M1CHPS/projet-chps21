#include "BatchLocation.hpp"


namespace nnet {
  BatchLocation::BatchLocation(const std::vector<math::clFTensor> &inputs,
                               const std::vector<math::clFTensor> &targets, size_t tensor_index,
                               size_t local_index)
      : inputs(&inputs), targets(&targets), tensor_index(tensor_index), local_index(local_index) {}

  void BatchLocation::progress(size_t count) {
    local_index += count;
    while (local_index > 0 and local_index >= (*inputs)[tensor_index].getDepth()) {
      local_index -= (*inputs)[tensor_index].getDepth();
      tensor_index++;
      // If we have reached the end of the input sets, loop back to the beginning
      if (tensor_index >= inputs->size()) tensor_index = 0;
    }
  }

}   // namespace nnet