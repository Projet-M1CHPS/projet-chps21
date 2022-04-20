#pragma once
#include "Controller.hpp"
#include "InputSet.hpp"
#include "NeuralNetwork.hpp"

namespace control {

  /**
   * @brief Evaluator that runs a model on a set of inputs and assign each sample its label
   * according to the output of the model.
   */
  class EvalController : Controller {
  public:
    EvalController(const std::filesystem::path &output_path, nnet::Model *model,
                   InputSet *input_set);

    /**
     * @brief Runs the model on the input set. This methods feeds the model with tensors inside
     * the input set for better performance. Tensors may be computed asynchronously.
     *
     * @param wrapper
     * @return
     */
    ControllerResult run() noexcept override;

  private:
    nnet::Model *model;
    InputSet *input_set;
  };

}   // namespace control
