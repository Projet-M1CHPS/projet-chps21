#include "MLPBatchOptimizer.hpp"

namespace nnet {

  MLPBatchOptimizer::MLPBatchOptimizer(MLPModel &model, std::unique_ptr<Optimization> tm)
      : MLPOptimizer(model, std::move(tm)) {
    auto &mlp_model = model;

    auto &perceptron = mlp_model.getPerceptron();
    auto &topology = perceptron.getTopology();

    storage = BackpropStorage(neural_network->getWeights());

    layers.resize(perceptron.getWeights().size() + 1);
    layers_af.resize(perceptron.getWeights().size() + 1);

    for (size_t i = 0; i < perceptron.getWeights().size(); i++) {
      math::FloatMatrix buf1(topology[i + 1], topology[i]);
      buf1.fill(0.0);
      avg_gradients.emplace_back(buf1);

      buf1 = math::FloatMatrix(topology[i + 1], 1);
      buf1.fill(0.0);
      avg_errors.emplace_back(buf1);
    }
  }

  void MLPBatchOptimizer::optimize(const std::vector<math::clFTensor> &inputs,
                                   const std::vector<math::clFTensor> &targets) {
    if (inputs.size() != targets.size())
      throw std::runtime_error("MLPBatchOptimizer: Inputs and targets number doesn't match !");

    size_t n = inputs.size();
    cl::CommandQueue queue =
            cl::CommandQueue(utils::cl_wrapper.getContext(), utils::cl_wrapper.getDefaultDevice());

    auto zerol = [&](auto &mat) { mat.fill(0.0f, queue); };
    std::for_each(avg_gradients.begin(), avg_gradients.end(), zerol);
    std::for_each(avg_errors.begin(), avg_errors.end(), zerol);


    for (long i = 0; i < n; i++) {
      auto input = inputs[i].flatten();
      auto target = targets[i].flatten();
      for (size_t j = 0; j < input.getZ(); j++) {
        forward(input.getMatrix(j), queue);
        computeGradient(queue);
      }
    }

    for (auto &it : avg_gradients) { it.ipscale(((float) 1.0 / static_cast<float>(n)), queue); }

    for (long i = neural_network->getWeights().size() - 1; i >= 0; i--) {
      storage.setIndex(i);
      std::swap(storage.getGradient(), avg_gradients[i]);
      std::swap(storage.getError(), avg_errors[i]);
      opti_meth->optimize(storage, queue);
      // Placeholder until we remove the backprop storage
      std::swap(avg_gradients[i], storage.getGradient());
      std::swap(avg_errors[i], storage.getError());
    }
    queue.finish();
  }


  void MLPBatchOptimizer::forward(math::clFMatrix const &inputs, cl::CommandQueue &queue) {
    auto &weights = this->neural_network->getWeights();
    auto &biases = this->neural_network->getBiases();
    auto &activation_functions = this->neural_network->getActivationFunctions();

    layers[0].copy(inputs, queue, false);
    layers_af[0].copy(inputs, queue, false);

    if (weights.empty()) return;

    math::clFMatrix current_layer =
            math::clFMatrix::gemm(1.0f, false, weights[0], false, inputs, 1.0f, biases[0], queue);
    layers[1].copy(current_layer, queue, false);
    af::applyAF(activation_functions[0], current_layer, queue);
    auto afunc = af::getAFFromType(activation_functions[0]).first;
    layers_af[1].copy(current_layer, queue, false);

    for (size_t k = 1; k < weights.size(); k++) {
      // C = W * C + B
      current_layer = math::clFMatrix::gemm(1.0f, false, weights[k], false, current_layer, 1.0f,
                                            biases[k], queue);
      layers[k + 1].copy(current_layer, queue, false);

      // Apply activation function on every element of the matrix
      af::applyAF(activation_functions[k], current_layer, queue);
      layers_af[k + 1].copy(current_layer, queue, false);
    }
  }

  void MLPBatchOptimizer::computeGradient(cl::CommandQueue &queue) {
    auto &weights = this->neural_network->getWeights();
    auto &biases = this->neural_network->getBiases();
    auto &activation_functions = this->neural_network->getActivationFunctions();

    if (weights.empty()) return;

    // Need to use a long since we stop when index reaches -1
    for (long i = weights.size() - 1; i >= 0; i--) {
      storage.setIndex(i);

      math::clFMatrix derivative;
      derivative.copy(layers[i + 1], queue, false);

      af::applyDerivativeAF(activation_functions[i], derivative, queue);

      derivative.iphadamard(storage.getError(), queue);

      storage.getError() = math::clFMatrix::gemm(1.0f, true, weights[i], false, derivative, queue);

      storage.getGradient() =
              math::clFMatrix::gemm(1.0f, false, derivative, true, layers_af[i], queue);

      avg_gradients[i].ipadd(1.0f, storage.getGradient(), queue);
    }
  }
}   // namespace nnet