#include "BatchOptimizationScheduler.hpp"

namespace nnet {

  BatchSchedulerJob::BatchSchedulerJob(size_t batch_size,
                                       const std::vector<math::clFTensor> &inputs,
                                       const std::vector<math::clFTensor> &targets)
      : batch_size(batch_size), inputs(&inputs), targets(&targets) {}

  size_t BatchSchedulerJob::getGlobalWorkSize() const {
    size_t res = 0;
    for (const auto &t : *inputs) { res += t.getDepth(); }
    return res;
  }

  BatchOptimizationScheduler::BatchOptimizationScheduler(const BatchSchedulerJob &job) : job(job) {}

}   // namespace nnet