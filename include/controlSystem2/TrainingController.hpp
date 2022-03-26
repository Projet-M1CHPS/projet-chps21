#pragma once
#include "Controller.hpp"
#include "NeuralNetwork.hpp"

namespace control {

  class TrainingController : Controller {
  public:
    TrainingController(std::filesystem::path const &output_path, nnet::Model *model,
                       nnet::Optimizer *optimizer, TrainingCollection *trainingCollection,
                       size_t max_epoch = 10, bool output_stats = true);

    ControllerResult run() override;

  private:

    nnet::Model* model;
    nnet::Optimizer* optimizer;
    TrainingCollection* training_collection;

    size_t max_epoch;
    bool is_outputting_stats;
  };

}   // namespace control
