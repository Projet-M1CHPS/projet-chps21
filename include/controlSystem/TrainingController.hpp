#pragma once
#include "Controller.hpp"
#include "ModelEvaluator.hpp"
#include "NeuralNetwork.hpp"
#include "TrainingCollection.hpp"

namespace control {

  class TrainingController {
  public:
    explicit TrainingController(size_t max_epoch, ModelEvolutionTracker &evaluator,
                                nnet::OptimizationScheduler &scheduler);

    virtual ControllerResult run();
    void setVerbose(bool v) { is_verbose = v; }
    bool isVerbose() const { return is_verbose; }

  protected:
    size_t max_epoch;
    bool is_verbose = false;
    ModelEvolutionTracker *evaluator;
    nnet::OptimizationScheduler *scheduler;
  };

}   // namespace control
