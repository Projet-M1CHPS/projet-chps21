#include "MLPStochOptimizer.hpp"

using namespace utils;
using namespace math;

namespace nnet {

  namespace {
    void applyAF(af::ActivationFunctionType type, math::clFMatrix &mat, utils::clWrapper &wrapper,
                 cl::CommandQueue &queue) {
      if (type == af::ActivationFunctionType::identity) return;
      auto afunc = af::getAFKernelFromType(type, wrapper).first;
      afunc.setArg(0, mat.getBuffer());
      queue.enqueueNDRangeKernel(afunc, cl::NullRange, mat.size(), cl::NullRange);
    }

    void applyDerivativeAF(af::ActivationFunctionType type, math::clFMatrix &mat,
                           utils::clWrapper &wrapper, cl::CommandQueue &queue) {
      if (type == af::ActivationFunctionType::identity) return;
      auto afunc = af::getAFKernelFromType(type, wrapper).second;
      afunc.setArg(0, mat.getBuffer());
      queue.enqueueNDRangeKernel(afunc, cl::NullRange, mat.size(), cl::NullRange);
    }
  }   // namespace


  MLPStochOptimizer::MLPStochOptimizer(MLPModel &model, std::shared_ptr<Optimization> tm)
      : MLPOptimizer(model, std::move(tm)) {
    auto &perceptron = model.getPerceptron();

    storage = BackpropStorage(perceptron.getWeights());
    layers.resize(perceptron.getWeights().size() + 1);
    layers_af.resize(perceptron.getWeights().size() + 1);
  }

  math::clFMatrix MLPStochOptimizer::optimize(const math::clFMatrix &input,
                                                const math::clFMatrix &target) {
    auto &weights = this->neural_network->getWeights();
    auto &topology = this->neural_network->getTopology();

    cl::CommandQueue queue(wrapper->getDefaultDevice());
    forward(input, queue);
    storage.getError() = layers_af[layers_af.size() - 1].sub(target, *wrapper);
    backward(target, queue);
    return {storage.getError(), *wrapper};
  }

  void MLPStochOptimizer::optimize(const std::vector<math::clFMatrix> &inputs,
                                   const std::vector<math::clFMatrix> &targets) {
    if (inputs.size() != targets.size()) {
      throw std::runtime_error("MLPModelStochOptimizer: Inputs and targets number doesn't match !");
    }

    for (size_t i = 0; i < inputs.size(); ++i) { optimize(inputs[i], targets[i]); }
  }


  void MLPStochOptimizer::forward(math::clFMatrix const &inputs, cl::CommandQueue &queue) {
    auto &weights = this->neural_network->getWeights();
    auto &biases = this->neural_network->getBiases();
    auto &activation_functions = this->neural_network->getActivationFunctions();

    layers[0] = clFMatrix(inputs, *wrapper, queue, false);
    layers_af[0] = clFMatrix(inputs, *wrapper, queue, false);

    if (weights.empty()) return;

    auto current_layer = clFMatrix::gemm(1.0f, false, weights[0], false, inputs, 1.0f, biases[0],
                                         *wrapper, queue);
    // Store the matrix before the activation function
    layers[1] = clFMatrix(current_layer, *wrapper, queue, false);

    // Apply the activation function
    applyAF(activation_functions[0], layers_af[1], *wrapper, queue);
    // Store the matrix after the activation function
    layers_af[1] = clFMatrix(current_layer, *wrapper, queue, false);

    for (size_t k = 1; k < weights.size(); k++) {
      // Multiply the current layer by the weights and add the biases
      current_layer = clFMatrix::gemm(1.0f, false, weights[k], false, current_layer, 1.0f,
                                      biases[k], *wrapper, queue);

      // Store the matrix before the activation function
      layers[k + 1] = clFMatrix(current_layer, *wrapper, queue, false);

      applyAF(activation_functions[k], current_layer, *wrapper, queue);
      // Store the matrix after the activation function
      layers_af[k + 1] = clFMatrix(current_layer, *wrapper, false);
    }
  }

  void MLPStochOptimizer::backward(math::clFMatrix const &target, cl::CommandQueue &queue) {
    auto &weights = this->neural_network->getWeights();
    auto &biases = this->neural_network->getBiases();
    auto &activation_functions = this->neural_network->getActivationFunctions();

    // storage.current_error = layers_af[layers_af.size() - 1] - target;

    for (long i = weights.size() - 1; i >= 0; i--) {
      storage.setIndex(i);

      math::clFMatrix derivative(layers[i + 1], *wrapper, queue, false);
      applyDerivativeAF(activation_functions[i], derivative, *wrapper, queue);

      derivative.hadamard(storage.getError(), *wrapper, queue);

      storage.getError() =
              clFMatrix::gemm(1.0f, true, weights[i], false, derivative, *wrapper, queue);

      storage.getGradient() =
              clFMatrix::gemm(1.0f, false, derivative, true, layers_af[i], *wrapper, queue);
      opti_meth->optimize(storage, *wrapper, queue);
    }
  }

}   // namespace nnet