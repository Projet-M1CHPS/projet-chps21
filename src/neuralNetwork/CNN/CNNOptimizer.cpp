#include "neuralNetwork/CNN/CNNOptimizer.hpp"


namespace nnet {

  CNNOptimizer::CNNOptimizer(CNNModel &model) : nn_cnn(&model.getCnn()), nn_mlp(&model.getMlp()) {}

  void CNNStochOptimizer::optimize(const math::clFMatrix &input, const math::clFMatrix &target) {
    clFMatrix flatten = forward(input);

    clFMatrix errorFlatten = mlp_opti.optimize(std::move(flatten), target);

    backward(input, errorFlatten);
  }

  void CNNStochOptimizer::optimize(const std::vector<math::clFMatrix> &inputs,
                                   const std::vector<math::clFMatrix> &targets) {
    if (inputs.size() != targets.size()) {
      throw std::runtime_error("Invalid number of inputs or targets");
    }
    for (size_t i = 0; i < inputs.size(); ++i) { optimize(inputs[i], targets[i]); }
  }

  clFMatrix CNNStochOptimizer::forward(const clFMatrix &input) {}

  void CNNStochOptimizer::backward(const clFMatrix &input, const clFMatrix &errorFlatten) {}

}   // namespace nnet