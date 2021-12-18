#pragma once
#include "classifierCollection.hpp"
#include "classifierInputSet.hpp"
#include "classifierTracker.hpp"
#include "controlSystem/controller.hpp"

namespace control::classifier {

  /** Classifier Training Controller, used for training a classifier model
   *
   */
  class CTController : public Controller {
  public:
    explicit CTController(TrainingControllerParameters params) : params(std::move(params)) {}

    /** Starts a run using the stored model and parameters
     *
     * On error, returns a ControllerResults object with the error set.
     * Guaranteed not to throw, even if it means catching unhandled exceptions.
     *
     * This behaviour is required to prevent an exception reaching the python layer
     *
     * @return
     */
    ControllerResult run() noexcept override;

    /** Starts a run using the stored model and parameters
     *
     * On error, returns a ControllerResults object with the error set.
     * Guaranteed not to throw, even if it means catching unhandled exceptions.
     *
     * This behaviour is required to prevent an exception reaching the python layer
     * Furthermore, appends a callback to the exit_handler to dump the model to disk if the program
     * unexpectedly terminates
     * @params e_handler ExitHandler for storing the model dump callback
     * @return
     */
    ControllerResult run(tscl::ExitHandler &e_handler) noexcept override;

  private:
    ControllerResult train();
    void trainingLoop(CTracker &stracker);
    void printPostTrainingStats(CTracker &stracker);

    TrainingControllerParameters params;
    std::shared_ptr<ClassifierTrainingCollection> training_collection;
  };
}   // namespace control::classifier