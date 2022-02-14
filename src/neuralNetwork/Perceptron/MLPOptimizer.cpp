
#include "MLPOptimizer.hpp"

namespace nnet {


  MLPOptimizer::MLPOptimizer(MLPModel &model, std::shared_ptr<Optimization> tm)
      : neural_network(&model.getPerceptron()), opti_meth(tm) {}


  MLPModelStochOptimizer::MLPModelStochOptimizer(MLPModel &model, std::shared_ptr<Optimization> tm)
      : MLPOptimizer(model, tm) {
    setModel(model);
  }


  void MLPModelStochOptimizer::train(const math::FloatMatrix &input,
                                     const math::FloatMatrix &target) {
    const size_t nbInput = input.getRows();
    const size_t nbTarget = target.getRows();

    auto &weights = this->neural_network->getWeights();
    auto &topology = this->neural_network->getTopology();

    if (nbInput != topology.getInputSize()) {
      throw std::runtime_error("Invalid number of inputs");
    }

    if (nbTarget != topology.getOutputSize()) throw std::runtime_error("Invalid number of targets");

    forward(input);
    storage.getError() = layers_af[layers_af.size() - 1] - target;
    backward(target);
  }

  void MLPModelStochOptimizer::setModel(MLPModel &model) {
    auto &mlp_model = dynamic_cast<MLPModel &>(model);
    auto &perceptron = mlp_model.getPerceptron();

    storage = BackpropStorage(perceptron.getWeights());
    layers.resize(perceptron.getWeights().size() + 1);
    layers_af.resize(perceptron.getWeights().size() + 1);
  }

  void MLPModelStochOptimizer::optimize(const std::vector<math::FloatMatrix> &inputs,
                                        const std::vector<math::FloatMatrix> &targets) {
    if (inputs.size() != targets.size()) { throw std::runtime_error("Invalid number of inputs"); }
    for (size_t i = 0; i < inputs.size(); ++i) { train(inputs[i], targets[i]); }
  }


  void MLPModelStochOptimizer::forward(math::FloatMatrix const &inputs) {
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

  void MLPModelStochOptimizer::backward(math::FloatMatrix const &target) {
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

      this->opti_meth->optimize(storage);
    }
  }


  MLPBatchOptimizer::MLPBatchOptimizer(MLPModel &model, std::shared_ptr<Optimization> tm)
      : MLPOptimizer(model, tm) {
    setModel(model);
  }


  void MLPBatchOptimizer::setModel(MLPModel &model) {
    auto &mlp_model = dynamic_cast<MLPModel &>(model);

    auto &perceptron = mlp_model.getPerceptron();
    auto &topology = perceptron.getTopology();

    storage = BackpropStorage(this->neural_network->getWeights());

    layers.resize(perceptron.getWeights().size() + 1);
    layers_af.resize(perceptron.getWeights().size() + 1);

    for (size_t i = 0; i < perceptron.getWeights().size(); i++) {
      avg_gradients.push_back(math::FloatMatrix(topology[i + 1], topology[i]));
      avg_errors.push_back(math::FloatMatrix(topology[i + 1], 1));

      avg_gradients[i].fill(0.0);
      avg_errors[i].fill(0.0);
    }
  }

  void MLPBatchOptimizer::optimize(const std::vector<math::FloatMatrix> &inputs,
                                   const std::vector<math::FloatMatrix> &targets) {
    size_t n = inputs.size();

    auto mat_reset = [](math::FloatMatrix &m) { m.fill(0); };
    std::for_each(avg_gradients.begin(), avg_gradients.end(), mat_reset);
    std::for_each(avg_errors.begin(), avg_errors.end(), mat_reset);

    for (long i = 0; i < n; i++) {
      forward(inputs[i]);
      storage.getError() = layers_af[layers_af.size() - 1] - targets[i];
      computeGradient();
    }

    for (auto &it : avg_gradients) { it *= ((float) 1.0 / n); }

    for (long i = this->neural_network->getWeights().size() - 1; i >= 0; i--) {
      storage.setIndex(i);
      storage.getGradient() = avg_gradients[i];
      storage.getError() = avg_errors[i];
      this->opti_meth->optimize(storage);
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

    // Avoid the loop index underflowing back to +inf
    if (weights.empty()) return;

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

  MLPMiniBatchOptimizer::MLPMiniBatchOptimizer(MLPModel &model, std::shared_ptr<Optimization> tm,
                                               size_t batch_size)
      : MLPBatchOptimizer(model, std::move(tm)), batch_size(batch_size) {}


  void MLPMiniBatchOptimizer::optimize(const std::vector<math::FloatMatrix> &inputs,
                                       const std::vector<math::FloatMatrix> &targets) {
    size_t n = inputs.size();

    for (size_t i = 0; i < n; i += batch_size) {
      auto mat_reset = [](math::FloatMatrix &m) { m.fill(0); };
      std::for_each(avg_gradients.begin(), avg_gradients.end(), mat_reset);
      std::for_each(avg_errors.begin(), avg_errors.end(), mat_reset);

      // If n is not a multiple of batch_size, the last batch will be smaller
      size_t curr_batch_size = std::min(batch_size, n - i);

      // Compute the average gradient and error for the current batch
      for (size_t j = 0; j < curr_batch_size; j++) {
        forward(inputs[i + j]);
        storage.getError() = layers_af[layers_af.size() - 1] - targets[i + j];
        computeGradient();
      }

      for (auto &it : avg_gradients) { it *= 1.0f / static_cast<float>(curr_batch_size); }

      // Update the weights and biases using the average of the current batch
      for (long j = this->neural_network->getWeights().size() - 1; j >= 0; j--) {
        storage.setIndex(j);
        storage.getGradient() = avg_gradients[j];
        storage.getError() = avg_errors[j];
        this->opti_meth->optimize(storage);
      }
    }
  }

}   // namespace nnet