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

    clFTensor backward(MLPerceptron &perceptron, const clFTensor &targets,
                       std::vector<clFTensor> &layers_output,
                       std::vector<clFTensor> &layers_af_output,
                       MLPOptimizer::WeightUpdateCache &updater, cl::CommandQueue &queue) {
      auto &weights = perceptron.getWeights();
      auto &activation_functions = perceptron.getActivationFunctions();

      if (weights.empty()) return {};

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
        updater.add(i, collapsed_gradient, gradient.getDepth(), queue);
      }
      return error;
    }
  }   // namespace

  using WeightUpdateCache = MLPOptimizer::WeightUpdateCache;

  WeightUpdateCache::WeightUpdateCache(MLPOptimizer &optimizer)
      : perceptron(optimizer.neural_network), optimization(optimizer.opti_meth.get()) {
    weight_updates.resize(perceptron->getWeights().size());
    contributions.resize(perceptron->getWeights().size(), 0);

    auto queue = utils::cl_wrapper.getDefaultQueue();
    for (size_t i = 0; auto &w : perceptron->getWeights()) {
      weight_updates[i] = clFMatrix(w.getRows(), w.getCols());
      weight_updates[i].fill(0.0f, queue, false);
      i++;
    }
    queue.finish();
  }

  void WeightUpdateCache::add(size_t index, const clFMatrix &delta, size_t contribution_size,
                              cl::CommandQueue &queue) {
    weight_updates[index].ipadd(1.0f, delta, queue);
    contributions[index] += contribution_size;
  }

  void WeightUpdateCache::reduce(WeightUpdateCache &other, cl::CommandQueue &queue) {
    for (size_t i = 0; i < weight_updates.size(); i++) {
      weight_updates[i].ipadd(1.0f, other[i], queue);
      contributions[i] += other.contributions[i];
    }
  }

  void WeightUpdateCache::clear(cl::CommandQueue &queue) {
    for (auto &w : weight_updates) { w.fill(0.0f, queue); }

    for (auto &c : contributions) { c = 0; }
  }

  void WeightUpdateCache::apply(cl::CommandQueue &queue) {
    for (size_t i = 0; i < weight_updates.size(); i++) {
      float mean_factor = 1.0f / static_cast<float>(contributions[i]);
      weight_updates[i].ipscale(mean_factor, queue);
      optimization->optimize(weight_updates[i], perceptron->getWeights()[i], i, queue);
      weight_updates[i].fill(0.0f, queue);
    }
  }


  std::unique_ptr<WeightUpdateCache> MLPOptimizer::makeCache(size_t ncache) {
    return std::make_unique<WeightUpdateCache>(*this);
  }

  std::vector<std::unique_ptr<WeightUpdateCache>> MLPOptimizer::makeCaches(size_t ncache) {
    std::vector<std::unique_ptr<WeightUpdateCache>> caches;
    caches.reserve(ncache);
    for (size_t i = 0; i < ncache; i++) {
      caches.emplace_back(makeCache(ncache));
    }
    return caches;
  }

  std::unique_ptr<Optimizer::Operation> MLPOptimizer::makeBatchOperation() {
    return std::make_unique<Operation>(*this);
  }

  clFTensor MLPOptimizer::optimize(const clFTensor &inputs, const clFTensor &targets,
                                   WeightUpdateCache &updater, cl::CommandQueue &queue) {
    std::vector<clFTensor> layers_output(neural_network->getWeights().size() + 1);
    std::vector<clFTensor> layers_af_output(neural_network->getWeights().size() + 1);

    forward(*neural_network, inputs.flatten(), layers_output, layers_af_output, queue);
    return backward(*neural_network, targets.flatten(), layers_output, layers_af_output, updater,
                    queue);
  }

  math::clFTensor MLPOptimizer::Operation::computeGradient(size_t thread_rank,
                                                           const math::clFTensor &inputs,
                                                           const math::clFTensor &targets,
                                                           cl::CommandQueue &batch_queue) {
    if (thread_rank > caches.size())
      throw std::invalid_argument("Error: Only " + std::to_string(caches.size()) +
                                  " caches reserved, tried to access cache " +
                                  std::to_string(thread_rank));
    return optimizer->optimize(inputs, targets, *caches[thread_rank], batch_queue);
  }


  void MLPOptimizer::Operation::reserveCaches(size_t num_threads) {
    if (caches.size() < num_threads) { caches = optimizer->makeCaches(num_threads); }
  }

  void MLPOptimizer::Operation::reduceAll(cl::CommandQueue &queue) {
    for (size_t i = 1; i < caches.size(); i++) { caches[0]->reduce(*caches[i], queue); }
  }

  void MLPOptimizer::Operation::applyChanges(cl::CommandQueue &queue) { caches[0]->apply(queue); }

  void MLPOptimizer::Operation::clearChanges(cl::CommandQueue &queue) {
    for (auto &cache : caches) { cache->clear(queue); }
  }
}   // namespace nnet
