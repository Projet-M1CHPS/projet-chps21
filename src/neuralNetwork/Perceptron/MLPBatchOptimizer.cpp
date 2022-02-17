#include "MLPBatchOptimizer.hpp"

namespace nnet {

  MLPBatchOptimizer::MLPBatchOptimizer(MLPModel &model, std::shared_ptr<Optimization> tm)
      : MLPOptimizer(model, tm) {
    auto &mlp_model = model;

    auto &perceptron = mlp_model.getPerceptron();
    auto &topology = perceptron.getTopology();

    storage = BackpropStorage(neural_network->getWeights());

    layers.resize(perceptron.getWeights().size() + 1);
    layers_af.resize(perceptron.getWeights().size() + 1);

    for (size_t i = 0; i < perceptron.getWeights().size(); i++) {
      avg_gradients.emplace_back(topology[i + 1], topology[i]);
      avg_errors.emplace_back(topology[i + 1], 1);

      avg_gradients[i].fill(0.0);
      avg_errors[i].fill(0.0);
    }
  }

  void MLPBatchOptimizer::optimize(const std::vector<math::FloatMatrix> &inputs,
                                   const std::vector<math::FloatMatrix> &targets) {
    if (inputs.size() != targets.size())
      throw std::runtime_error("MLPBatchOptimizer: Inputs and targets number doesn't match !");

    size_t n = inputs.size();
    auto mat_reset = [](math::FloatMatrix &m) { m.fill(0); };
    std::for_each(avg_gradients.begin(), avg_gradients.end(), mat_reset);
    std::for_each(avg_errors.begin(), avg_errors.end(), mat_reset);

    for (long i = 0; i < n; i++) {
      forward(inputs[i]);
      storage.getError() = layers_af[layers_af.size() - 1] - targets[i];
      computeGradient();
    }

    for (auto &it : avg_gradients) { it *= ((float) 1.0 / static_cast<float>(n)); }

    for (long i = neural_network->getWeights().size() - 1; i >= 0; i--) {
      storage.setIndex(i);
      storage.getGradient() = avg_gradients[i];
      storage.getError() = avg_errors[i];
      opti_meth->optimize(storage);
    }
  }


  void MLPBatchOptimizer::forward(math::FloatMatrix const &inputs) {
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

  void MLPBatchOptimizer::computeGradient() {
    auto &weights = this->neural_network->getWeights();
    auto &biases = this->neural_network->getBiases();
    auto &activation_functions = this->neural_network->getActivationFunctions();

    if (weights.empty()) return;

    // Need to use a long since we stop when index reaches -1
    for (long i = weights.size() - 1; i >= 0; i--) {
      storage.setIndex(i);

      math::FloatMatrix derivative(layers[i + 1]);
      auto dafunc = af::getAFFromType(activation_functions[i]).second;
      std::transform(derivative.cbegin(), derivative.cend(), derivative.begin(), dafunc);

      derivative.hadamardProd(storage.getError());

      storage.getError() = math::FloatMatrix::mul(true, weights[i], false, derivative);

      storage.getGradient() = math::FloatMatrix::mul(false, derivative, true, layers_af[i], 1.0);

      avg_gradients[i] += storage.getGradient();
    }
  }
}   // namespace nnet