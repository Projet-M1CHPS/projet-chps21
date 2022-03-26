#pragma once
#include "Controller.hpp"
#include "NeuralNetwork.hpp"

namespace control {

  class EvalController : Controller {
  public:
    EvalController(const std::filesystem::path &output_path, nnet::Model *model,
                   InputSet *input_set, bool output_stats = true);

    ControllerResult run() override;

  private:
    nnet::Model *model;
    InputSet *input_set;
    bool is_outputting_stats;
  };

}   // namespace control
