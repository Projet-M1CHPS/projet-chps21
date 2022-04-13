#pragma once
#include "Controller.hpp"
#include "NeuralNetwork.hpp"
#include "TrainingCollection.hpp"
#include "mpi.h"

namespace control {

  class MPITrainingController : Controller {
  public:
    MPITrainingController(std::filesystem::path const &output_path, nnet::Model &model,
                          nnet::Optimizer &optimizer, TrainingCollection &trainingCollection,
                          size_t max_epoch = 10, bool output_stats = true);

    ControllerResult run() override;

  private:
    nnet::Model *model;
    nnet::Optimizer *optimizer;
    TrainingCollection *training_collection;

    size_t max_epoch;
    bool is_outputting_stats;
  };

}   // namespace control
