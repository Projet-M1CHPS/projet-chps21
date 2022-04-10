#include "MLPOptimizer.hpp"

using namespace math;

namespace nnet {

  namespace {
    void forward(MLPerceptron &perceptron, const clFTensor &inputs,
                 std::vector<clFTensor> &layers_output, std::vector<clFTensor> &layers_af_output,
                 cl::CommandQueue &queue) {
      auto &weights = perceptron.getWeights();
      auto &biases = perceptron.getBiases();
      auto &activation_functions = perceptron.getActivationFunctions();

      layers_output[0].copy(inputs, queue, false);
      layers_af_output[0].copy(inputs, queue, false);

      if (weights.empty()) return;

      clFTensor current_layer = clFTensor::batchedGemm(1.0f, false, weights[0], false, inputs, 1.0f,
                                                       biases[0], queue);

      layers_output[1].copy(current_layer, queue, false);
      af::applyAF(activation_functions[0], current_layer, queue);
      layers_af_output[1].copy(current_layer, queue, false);

      for (size_t k = 1; k < weights.size(); k++) {
        // C = W * C + B
        current_layer = clFTensor::batchedGemm(1.0f, false, weights[k], false, current_layer, 1.0f,
                                               biases[k], queue);
        layers_output[k + 1].copy(current_layer, queue, false);

        // Apply activation function on every element of the matrix
        af::applyAF(activation_functions[k], current_layer, queue);
        layers_af_output[k + 1].copy(current_layer, queue, false);
      }
    }

    void backward(MLPerceptron &perceptron, const clFTensor &targets,
                  std::vector<clFTensor> &layers_output, std::vector<clFTensor> &layers_af_output,
                  MLPWeightUpdater &updater, cl::CommandQueue &queue) {
      auto &weights = perceptron.getWeights();
      auto &biases = perceptron.getBiases();
      auto &activation_functions = perceptron.getActivationFunctions();

      if (weights.empty()) return;

      clFTensor error = layers_af_output.back().sub(1.0f, targets, queue);

      // Need to use a long since we stop when index reaches -1
      for (long i = weights.size() - 1; i >= 0; i--) {
        clFTensor derivative;
        derivative.copy(layers_output[i + 1], queue, false);
        af::applyDerivativeAF(activation_functions[i], derivative, queue);
        derivative.iphadamard(error, queue);
        error = clFTensor::batchedGemm(1.0f, true, weights[i], false, derivative, queue);

        clFTensor gradient =
                clFTensor::batchedGemm(1.0f, false, derivative, true, layers_af_output[i], queue);

        // Reduce the gradient to a single matrix
        clFMatrix collapsed_gradient = gradient.sumCollapse(queue);
        // The reducer cannot proceed until the gradient is collapsed
        // So we create a new event that the reducer can wait on
        cl::Event reduce_event;
        queue.enqueueMarkerWithWaitList(nullptr, &reduce_event);
        updater.reduce(i, collapsed_gradient, gradient.getDepth(), reduce_event);
      }
    }
  }   // namespace


  MLPWeightUpdater::MLPWeightUpdater(MLPerceptron &parent, Optimization &optimization)
      : perceptron(&parent), optimization(&optimization) {
    weight_updates.resize(parent.getWeights().size());
    contributions.resize(parent.getWeights().size(), 0);

    work_queue =
            cl::CommandQueue(utils::cl_wrapper.getContext(), utils::cl_wrapper.getDefaultDevice());
    for (size_t i = 0; auto &w : parent.getWeights()) {
      weight_updates[i] = clFMatrix(w.getRows(), w.getCols());
      weight_updates[i].fill(0.0f, work_queue);
      i++;
    }
    work_queue.finish();
  }

  const clFMatrix &MLPWeightUpdater::operator[](size_t i) {
    if (i > weight_updates.size()) {
      throw std::out_of_range("MLPWeightUpdater::operator[]: Index out of range");
    }
    return weight_updates[i];
  }

  void MLPWeightUpdater::reduce(size_t index, const clFMatrix &delta, size_t contribution_size,
                                cl::Event &event) {
    std::vector<cl::Event> wait_list = {event};
    work_queue.enqueueBarrierWithWaitList(&wait_list);
    weight_updates[index].ipadd(1.0f, delta, work_queue);
    std::scoped_lock<std::mutex> lock(mutex);
    contributions[index] += contribution_size;
  }

  void MLPWeightUpdater::apply() {
    for (size_t i = 0; i < weight_updates.size(); i++) {
      float mean_factor = 1.0f / static_cast<float>(contributions[i]);
      weight_updates[i].ipscale(mean_factor, work_queue);
      optimization->optimize(weight_updates[i], perceptron->getWeights()[i], i, work_queue);
      weight_updates[i].fill(0.0f, work_queue);
    }
    std::for_each(contributions.begin(), contributions.end(), [](auto &c) { c = 0; });
    work_queue.finish();
  }

  std::unique_ptr<OptimizerOperation> MLPOptimizer::makeBatchOperation() {
    auto updater = std::make_shared<MLPWeightUpdater>(*neural_network, *opti_meth);
    return std::make_unique<MLPBatchOperation>(*this, updater);
  }

  void MLPOptimizer::optimize(const clFTensor &inputs, const clFTensor &targets,
                              MLPWeightUpdater &updater, cl::CommandQueue &queue) {
    std::vector<clFTensor> layers_output(neural_network->getWeights().size() + 1);
    std::vector<clFTensor> layers_af_output(neural_network->getWeights().size() + 1);

    forward(*neural_network, inputs.flatten(), layers_output, layers_af_output, queue);
    backward(*neural_network, targets.flatten(), layers_output, layers_af_output, updater, queue);
  }

}   // namespace nnet
