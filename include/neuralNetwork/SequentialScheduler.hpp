#pragma once
#include "OptimizationBatchScheduler.hpp"

namespace nnet {
  class SequentialScheduler : public OptimizationBatchScheduler {
    ParallelScheduler(size_t batch_size, Optimizer &optimizer, const ParallelPolicy &policy);

    const utils::ParallelPolicy &getPolicy() const;

    void run() override;

  protected:

    void updateModel() override;
    void print(std::ostream &os) const override;

    void epochStart() override{};
    void endEpoch() override{};

  };
}   // namespace nnet
