#pragma once
#include "OptimizationScheduler.hpp"

namespace nnet {
  class SchedulerProfiler : public SchedulerDecorator {
  public:
    using SchedulerDecorator::SchedulerDecorator;

    /**
     * @brief Run the optimization process, using all the resources available in the scheduler.
     */
    void run() override;

  protected:
    /**
     * @brief Update the model after a batch of training.
     */
    void updateModel() override;

    /**
     * @brief Called at the start of the epoch. Can be used for pre-processing, to reset the state
     * of the scheduler, etc. The scheduler is free to ignore this call.
     */
    void epochStart() override;

    /**
     * @brief Called at the end of the epoch. Can be used for post-processing, to reset the state of
     * the scheduler, etc. The scheduler is free to ignore this call.
     */
    void endEpoch() override;

    /**
     * @brief Helper method to print the scheduler.
     * @param os
     */
    void print(std::ostream &os) const override;
  };
}   // namespace nnet
