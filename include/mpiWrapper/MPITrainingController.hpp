#pragma once
#include "Controller.hpp"
#include "ModelEvaluator.hpp"
#include "NeuralNetwork.hpp"
#include "TrainingCollection.hpp"
#include "TrainingController.hpp"
#include <mpi.h>

namespace control {

  class MPITrainingController : public TrainingController {
  public:
    explicit MPITrainingController(size_t maxEpoch, ModelEvolutionTracker &evaluator,
                                   nnet::OptimizationScheduler &scheduler);
    ControllerResult run() override;
  };
}   // namespace control
