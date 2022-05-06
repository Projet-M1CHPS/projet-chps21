#pragma once
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace nnet {

  /**
   * @brief Interface class for all scheduler. Allows to wrap and schedule the optimization process.
   */
  class OptimizationScheduler {
  public:
    friend std::ostream &operator<<(std::ostream &os, const OptimizationScheduler &scheduler) {
      scheduler.print(os);
      return os;
    }

    virtual ~OptimizationScheduler() = default;

    /**
     * @brief Run the optimization process, using all the resources available in the scheduler.
     */
    virtual void run() = 0;

  protected:
    // TODO Refactor me!
    // Implementing those methods as protected, and not implementing them in as a non-virtual method
    // provides no guarantee that the scheduler will be used/inherited correctly
    // (e.g an inherited scheduler might not call epochStart or endEpoch)
    // To enforce : make run() non-virtual and call each of the following methods
    // This forces inheritance of the scheduler to call the right methods

    /**
     * @brief Update the model after a batch of training.
     */
    virtual void updateModel() = 0;

    /**
     * @brief Called at the start of the epoch. Can be used for pre-processing, to reset the state
     * of the scheduler, etc. The scheduler is free to ignore this call.
     */
    virtual void epochStart() = 0;

    /**
     * @brief Called at the end of the epoch. Can be used for post-processing, to reset the state of
     * the scheduler, etc. The scheduler is free to ignore this call.
     */
    virtual void endEpoch() = 0;

    /**
     * @brief Helper method to print the scheduler.
     * @param os
     */
    virtual void print(std::ostream &os) const = 0;
  };

  /**
   * @brief Helper class to write decorators on the scheduler.
   */
  class SchedulerDecorator : public OptimizationScheduler {
  public:
    explicit SchedulerDecorator(std::shared_ptr<OptimizationScheduler> wrappee)
        : wrappee(std::move(wrappee)) {}

  private:
    std::shared_ptr<OptimizationScheduler> wrappee;
  };

}   // namespace nnet
