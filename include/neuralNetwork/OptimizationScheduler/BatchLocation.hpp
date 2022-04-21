#pragma once
#include "math/clFTensor.hpp"
#include <vector>

namespace nnet {

  /**
   * @brief Stores the current location inside a batch job, and provides facility to retrieve the
   * next tensor
   */
  class BatchLocation {
  public:
    /**
     * @brief Creates a location at the beginning of the batch
     * @param inputs The inputs of the batch
     * @param targets The associated targets
     * @param tensor_index The starting tensor index
     * @param local_index The starting index inside the tensor
     */
    BatchLocation(const std::vector<math::clFTensor> &inputs,
                  const std::vector<math::clFTensor> &targets, size_t tensor_index = 0,
                  size_t local_index = 0);

    /**
     * @brief Progresses the location by a number of elements
     * @param count
     */
    void progress(size_t count);

    /**
     * @brief Returns the number of remaining elements in the current tensors
     * @return
     */
    size_t getBatchRemainder() const { return (*inputs)[tensor_index].getDepth() - local_index; }

    /**
     * @brief Returns a slice of the next input tensor
     * @param size The size of the slice. Cannot be larger than the number of remaining elements
     * @return
     */
    math::clFTensor getInputSlice(size_t size) const {
      return (*inputs)[tensor_index].slice(local_index, local_index + size);
    }

    /**
     * @brief Returns a slice of the next target tensor
     * @param size The size of the slice. Cannot be larger than the number of remaining elements
     * @return
     */
    math::clFTensor getTargetSlice(size_t size) const {
      return (*targets)[tensor_index].slice(local_index, local_index + size);
    }

  private:
    const std::vector<math::clFTensor> *inputs, *targets;
    size_t tensor_index, local_index;
  };

}   // namespace nnet
