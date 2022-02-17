#include "MLPStochOptimizer.hpp"

namespace nnet {
  MLPStochOptimizer::MLPStochOptimizer(MLPModel &model, std::shared_ptr<Optimization> tm)
      : MLPOptimizer(model, std::move(tm)) {
    auto &perceptron = model.getPerceptron();

    storage = BackpropStorage(perceptron.getWeights());
    layers.resize(perceptron.getWeights().size() + 1);
    layers_af.resize(perceptron.getWeights().size() + 1);
  }

  math::FloatMatrix MLPStochOptimizer::optimize(const math::FloatMatrix &input,
                                                const math::FloatMatrix &target) {
    auto &weights = this->neural_network->getWeights();
    auto &topology = this->neural_network->getTopology();

    forward(input);
    storage.getError() = layers_af[layers_af.size() - 1] - target;
    backward(target);
    return storage.getError();
  }

  void MLPStochOptimizer::optimize(const std::vector<math::FloatMatrix> &inputs,
                                   const std::vector<math::FloatMatrix> &targets) {
    if (inputs.size() != targets.size()) {
      throw std::runtime_error("MLPModelStochOptimizer: Inputs and targets number doesn't match !");
    }

    for (size_t i = 0; i < inputs.size(); ++i) { optimize(inputs[i], targets[i]); }
  }


  void MLPStochOptimizer::forward(math::FloatMatrix const &inputs) {
    auto &weights = this->neural_network->getWeights();
    auto &biases = this->neural_network->getBiases();
    auto &activation_functions = this->neural_network->getActivationFunctions();

    layers[0] = inputs;
    layers_af[0] = inputs;

    if (weights.empty()) return;

    math::FloatMatrix current_layer =
            math::FloatMatrix::matMatProdMatAdd(weights[0], inputs, biases[0]);
    layers[1] = current_layer;
    auto afunc = af::getAFFromType(activation_functions[0]).first;
    std::transform(current_layer.cbegin(), current_layer.cend(), current_layer.begin(), afunc);
    layers_af[1] = current_layer;

    for (size_t k = 1; k < weights.size(); k++) {
      // C = W * C + B
      current_layer = math::FloatMatrix::matMatProdMatAdd(weights[k], current_layer, biases[k]);
      layers[k + 1] = current_layer;

      // Apply activation function on every element of the matrix
      afunc = af::getAFFromType(activation_functions[k]).first;
      std::transform(current_layer.cbegin(), current_layer.cend(), current_layer.begin(), afunc);

      layers_af[k + 1] = current_layer;
    }
  }

  void MLPStochOptimizer::backward(math::FloatMatrix const &target) {
    auto &weights = this->neural_network->getWeights();
    auto &biases = this->neural_network->getBiases();
    auto &activation_functions = this->neural_network->getActivationFunctions();

    // storage.current_error = layers_af[layers_af.size() - 1] - target;

    for (long i = weights.size() - 1; i >= 0; i--) {
      storage.setIndex(i);

      math::FloatMatrix derivative(layers[i + 1]);
      auto dafunc = af::getAFFromType(activation_functions[i]).second;
      std::transform(derivative.cbegin(), derivative.cend(), derivative.begin(), dafunc);

      derivative.hadamardProd(storage.getError());

      storage.getError() = math::FloatMatrix::mul(true, weights[i], false, derivative);

      storage.getGradient() = math::FloatMatrix::mul(false, derivative, true, layers_af[i], 1.0);

      opti_meth->optimize(storage);
    }
  }

}   // namespace nnet