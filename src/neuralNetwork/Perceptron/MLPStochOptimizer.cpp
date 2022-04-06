#include "MLPStochOptimizer.hpp"

using namespace utils;
using namespace math;

namespace nnet {

  MLPStochOptimizer::MLPStochOptimizer(MLPModel &model, std::unique_ptr<Optimization> tm)
      : MLPOptimizer(model, std::move(tm)) {
    auto &perceptron = model.getPerceptron();

    storage = BackpropStorage(perceptron.getWeights());
    layers.resize(perceptron.getWeights().size() + 1);
    layers_af.resize(perceptron.getWeights().size() + 1);
  }

  MLPStochOptimizer::MLPStochOptimizer(MLPerceptron &perceptron, std::unique_ptr<Optimization> tm)
      : MLPOptimizer(perceptron, std::move(tm)) {
    storage = BackpropStorage(perceptron.getWeights());
    layers.resize(perceptron.getWeights().size() + 1);
    layers_af.resize(perceptron.getWeights().size() + 1);
  }

  math::clFMatrix MLPStochOptimizer::optimize(const math::clFMatrix &input,
                                              const math::clFMatrix &target) {
    auto &weights = this->neural_network->getWeights();
    auto &topology = this->neural_network->getTopology();

    // cl::CommandQueue queue(wrapper->getContext(), wrapper->getDefaultDevice());
    auto &queue = utils::cl_wrapper.getDefaultQueue();
    forward(input, queue);
    storage.getError() = layers_af[layers_af.size() - 1].sub(1.0f, target, queue);
    backward(target, queue);

    clFMatrix res;
    res.copy(storage.getError(), queue, false);
    return res;
  }

  void MLPStochOptimizer::optimize(const std::vector<math::clFTensor> &inputs,
                                   const std::vector<math::clFTensor> &targets) {
    if (inputs.size() != targets.size()) {
      throw std::runtime_error("MLPModelStochOptimizer: Inputs and targets number doesn't match !");
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
      auto tensor = inputs[i].flatten();
      auto target = targets[i].flatten();
      for (size_t j = 0; j < inputs[i].getZ(); ++j) {
        optimize(tensor.getMatrix(j), target.getMatrix(j));
      }
    }
    utils::cl_wrapper.getDefaultQueue().finish();
  }


  void MLPStochOptimizer::forward(math::clFMatrix const &inputs, cl::CommandQueue &queue) {
    auto &weights = this->neural_network->getWeights();
    auto &biases = this->neural_network->getBiases();
    auto &activation_functions = this->neural_network->getActivationFunctions();

    layers[0].copy(inputs, queue, false);
    layers_af[0].copy(inputs, queue, false);

    if (weights.empty()) return;

    auto current_layer =
            clFMatrix::gemm(1.0f, false, weights[0], false, inputs, 1.0f, biases[0], queue);

    // Store the matrix before the activation function
    layers[1].copy(current_layer, queue, false);

    // Apply the activation function
    af::applyAF(activation_functions[0], current_layer, queue);
    // Store the matrix after the activation function
    layers_af[1].copy(current_layer, queue, false);

    for (size_t k = 1; k < weights.size(); k++) {
      // Multiply the current layer by the weights and add the biases
      current_layer = clFMatrix::gemm(1.0f, false, weights[k], false, current_layer, 1.0f,
                                      biases[k], queue);

      // Store the matrix before the activation function
      layers[k + 1].copy(current_layer, queue, false);

      af::applyAF(activation_functions[k], current_layer, queue);
      // Store the matrix after the activation function
      layers_af[k + 1].copy(current_layer, queue, false);
    }
  }

  void MLPStochOptimizer::backward(math::clFMatrix const &target, cl::CommandQueue &queue) {
    auto &weights = this->neural_network->getWeights();
    auto &biases = this->neural_network->getBiases();
    auto &activation_functions = this->neural_network->getActivationFunctions();

    for (long i = weights.size() - 1; i >= 0; i--) {
      storage.setIndex(i);

      math::clFMatrix derivative;
      derivative.copy(layers[i + 1], queue, false);

      af::applyDerivativeAF(activation_functions[i], derivative, queue);

      derivative.iphadamard(storage.getError(), queue);

      storage.getError() = clFMatrix::gemm(1.0f, true, weights[i], false, derivative, queue);

      storage.getGradient() = clFMatrix::gemm(1.0f, false, derivative, true, layers_af[i], queue);
      opti_meth->optimize(storage, queue);
    }
  }

}   // namespace nnet