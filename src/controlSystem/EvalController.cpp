#include "EvalController.hpp"

namespace control {

  EvalController::EvalController(const std::filesystem::path &output_path, nnet::Model *model,
                                 InputSet *input_set)
      : Controller(output_path), model(model), input_set(input_set) {}


  ControllerResult EvalController::run() noexcept {
    if (input_set->getTensorCount() == 0) { return {1, "No input data found"}; }

    size_t count = 0;

    // Temporary implementation that treat each matrix individually and not tensor-wise
    for (auto &it : *input_set) {
      auto res = model->predict(utils::cl_wrapper.getDefaultQueue(), it.getData());
      auto class_id = res.imax();
      (*input_set)[count].setClass(class_id);
      std::cout << "Sample " << it.getId() << " is of class " << class_id << std::endl;
    }

    return {0, "Evaluation Success"};
  }
}   // namespace control