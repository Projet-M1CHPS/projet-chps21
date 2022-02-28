#include "MLPBatchOptimizer.hpp"

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

  MLPBatchOptimizer::MLPBatchOptimizer(MLPModel &model, std::shared_ptr<Optimization> tm)
      : MLPOptimizer(model, tm) {
    auto &mlp_model = model;

    auto &perceptron = mlp_model.getPerceptron();
    auto &topology = perceptron.getTopology();

    storage = BackpropStorage(neural_network->getWeights());

    layers.resize(perceptron.getWeights().size() + 1);
    layers_af.resize(perceptron.getWeights().size() + 1);

    for (size_t i = 0; i < perceptron.getWeights().size(); i++) {
      math::FloatMatrix buf1(topology[i], topology[i + 1]);
      buf1.fill(0.0);
      avg_gradients.emplace_back(buf1, *wrapper);

      buf1 = math::FloatMatrix(topology[i + 1], 1);
      buf1.fill(0.0);
      avg_errors.emplace_back(buf1, *wrapper);
    }
  }

  void MLPBatchOptimizer::optimize(const std::vector<math::clFMatrix> &inputs,
                                   const std::vector<math::clFMatrix> &targets) {
    if (inputs.size() != targets.size())
      throw std::runtime_error("MLPBatchOptimizer: Inputs and targets number doesn't match !");

    size_t n = inputs.size();
    cl::CommandQueue queue(wrapper->getDefaultDevice());
    cl::Kernel zero_out = wrapper->getKernels().getKernel("utils.cl", "zero_out");

    auto zerol = [&](auto &mat) {
      zero_out.setArg(0, mat);
      queue.enqueueNDRangeKernel(zero_out, cl::NullRange,
                                 cl::NDRange(mat.getRows(), mat.getCols()));
    };
    std::for_each(avg_gradients.begin(), avg_gradients.end(), zerol);
    std::for_each(avg_errors.begin(), avg_errors.end(), zerol);


    for (long i = 0; i < n; i++) {
      forward(inputs[i], queue);
      storage.getError() = layers_af[layers_af.size() - 1].sub(targets[i], *wrapper, queue);
      computeGradient(queue);
    }

    queue.finish();

    for (auto &it : avg_gradients) {
      it.ipscale(((float) 1.0 / static_cast<float>(n)), *wrapper, queue);
    }

    for (long i = neural_network->getWeights().size() - 1; i >= 0; i--) {
      storage.setIndex(i);
      std::swap(storage.getGradient(), avg_gradients[i]);
      std::swap(storage.getError(), avg_errors[i]);
      opti_meth->optimize(storage, *wrapper, queue);
    }
    queue.finish();
  }


  void MLPBatchOptimizer::forward(math::clFMatrix const &inputs, cl::CommandQueue &queue) {
    auto &weights = this->neural_network->getWeights();
    auto &biases = this->neural_network->getBiases();
    auto &activation_functions = this->neural_network->getActivationFunctions();

    layers[0] = math::clFMatrix(inputs, *wrapper, queue, false);
    layers_af[0] = math::clFMatrix(inputs, *wrapper, queue, false);

    if (weights.empty()) return;

    math::clFMatrix current_layer = math::clFMatrix::gemm(1.0f, false, weights[0], false, inputs,
                                                          1.0f, biases[0], *wrapper, queue);
    layers[1] = math::clFMatrix(current_layer, *wrapper, queue, false);
    applyAF(activation_functions[0], current_layer, *wrapper, queue);
    auto afunc = af::getAFFromType(activation_functions[0]).first;
    layers_af[1] = math::clFMatrix(current_layer, *wrapper, queue, false);

    for (size_t k = 1; k < weights.size(); k++) {
      // C = W * C + B
      current_layer = math::clFMatrix::gemm(1.0f, false, weights[k], false, current_layer, 1.0f,
                                            biases[k], *wrapper, queue);
      layers[k + 1] = math::clFMatrix(current_layer, *wrapper, queue, false);

      // Apply activation function on every element of the matrix
      applyAF(activation_functions[k], current_layer, *wrapper, queue);
      layers_af[k + 1] = math::clFMatrix(current_layer, *wrapper, queue, false);
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

      math::clFMatrix derivative(layers[i + 1], *wrapper, queue);
      applyDerivativeAF(activation_functions[i], derivative, *wrapper, queue);

      derivative.hadamard(storage.getError(), *wrapper, queue);

      storage.getError() =
              math::clFMatrix::gemm(1.0f, true, weights[i], false, derivative, *wrapper, queue);

      storage.getGradient() =
              math::clFMatrix::gemm(1.0f, false, derivative, true, layers_af[i], *wrapper, queue);

      avg_gradients[i].ipadd(storage.getGradient(), *wrapper, queue);
    }
  }
}   // namespace nnet