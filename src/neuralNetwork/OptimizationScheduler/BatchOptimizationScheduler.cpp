#include "BatchOptimizationScheduler.hpp"

namespace nnet {

  BatchOptimizationScheduler::BatchOptimizationScheduler(
          size_t batch_size, Optimizer &optimizer, const std::vector<math::clFTensor> &inputs,
          const std::vector<math::clFTensor> &targets)
      : batch_size(batch_size), input_tensors(&inputs), target_tensors(&targets),
        optimizer_operation(optimizer.makeBatchOperation()), optimizer(&optimizer) {}

}   // namespace nnet