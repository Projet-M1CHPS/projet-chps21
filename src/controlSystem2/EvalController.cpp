#include "EvalController.hpp"

namespace control {

  EvalController::EvalController(const std::filesystem::path &output_path, nnet::Model *model,
                                 InputSet *input_set)
      : Controller(output_path), model(model), input_set(input_set) {}


  ControllerResult EvalController::run() {
    if (input_set->getTensorCount() == 0) { return {1, "No input data found"}; }

    size_t count = 0;

    for (auto it : *input_set) {
      auto res = model->predict(it.getData()).toFloatMatrix();
      auto class_id = std::distance(res.begin(), std::max_element(res.begin(), res.end()));
      (*input_set)[count].setClass(class_id);
      std::cout << "Sample " << it.getId() << " is of class " <<class_id << std::endl;
    }

    /*
    // Start an async job on the first tensor
    std::future<math::clFTensor> future = std::async([this]() { return
    model->predict(*input_set->beginTensor()); });

    // Iterate over the rest of the tensors
    for (auto input = input_set->beginTensor()++; input != input_set->endTensor(); input++) {
      // Feed every tensor to the model

      // Wait until the current batch is finished
      auto results = future.get();

      // Compute the next batch while we treat the current one
      future = std::async(std::launch::async, [&input, this]() { return model->predict(*input); });

      // Assign each sample to its class
      for (size_t z = 0; z < results.getZ(); z++) {
        auto cl_matrix = results.getMatrix(z);
        auto matrix = cl_matrix.toFloatMatrix(wrapper);
        // Find the max value in the result matrix
        auto class_id =
                std::distance(matrix.begin(), std::max_element(matrix.begin(), matrix.end()));
        (*input_set)[count].setClass(class_id);
      }
    } */
    return {0, "Evaluation Success"};
  }
}   // namespace control