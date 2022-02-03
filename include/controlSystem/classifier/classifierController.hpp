#pragma once
#include "CTCollection.hpp"
#include "classifierInputSet.hpp"
#include "classifierTracker.hpp"
#include "controlSystem/controller.hpp"

namespace control::classifier {

  /** @brief Classifier training RunController, used for training a classifier Model
   *
   */
  class CTController : public RunController {
  public:
    /** Constructs a controller to train a classifier model
     * The controller doesn't assume ownership of any of anything besides its own parameters
     * Meaning the model, optimizer, and training collection must be kept alive for the lifetime of
     * the controller
     *
     * @param params The controller parameters
     * @param model The model to train
     * @param optimizer The optimizer used for training the model
     * @param collection The collection of input used for training
     */
    explicit CTController(const TrainingControllerParameters &params, nnet::Model &model,
                          nnet::ModelOptimizer &optimizer, CTCollection &collection)
        : RunController(model), params(params), optimizer(&optimizer),
          training_collection(&collection) {}

    /** Starts a run using the stored model, optimizer, training collection and parameters
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
     * FIXME: implement me
     *
     * This behaviour is required to prevent an exception reaching the python layer
     * Furthermore, appends a callback to the exit_handler to dump the model to disk if the program
     * unexpectedly terminates
     * @params e_handler ExitHandler for storing the model dump callback
     * @return
     */
    // ControllerResult run(tscl::ExitHandler &e_handler) noexcept override;

  private:
    ControllerResult train();
    void trainingLoop(CTracker &stracker);
    void printPostTrainingStats(CTracker &stracker);

    nnet::ModelOptimizer *optimizer;
    TrainingControllerParameters params;
    CTCollection *training_collection;
  };
}   // namespace control::classifier