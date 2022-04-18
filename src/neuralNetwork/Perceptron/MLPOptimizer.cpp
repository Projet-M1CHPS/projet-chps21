#include "MLPOptimizer.hpp"

using namespace math;

namespace nnet {

  namespace {
    void forward(MLPerceptron &perceptron, const clFTensor &inputs,
                 std::vector<clFTensor> &layers_output, std::vector<clFTensor> &layers_af_output,
                 MLPOptimizer::WeightUpdateCache &updater, cl::CommandQueue &queue) {
      auto &weights = updater.getWeightsCopy();
      auto &biases = updater.getBiasesCopy();
      auto &activation_functions = perceptron.getActivationFunctions();

      layers_output[0].copy(inputs, queue, false);
      layers_af_output[0].copy(inputs, queue, false);

      if (weights.empty()) return;


      clFTensor current_layer = clFTensor::batchedGemm(1.0f, false, weights[0], false, inputs, 1.0f,
                                                       biases[0], queue);
      // std::cout << "current_layer:\n" << current_layer;

      layers_output[1].copy(current_layer, queue, false);
      af::applyAF(activation_functions[0], current_layer, queue);
      layers_af_output[1].copy(current_layer, queue, false);
      // std::cout << "current_layer AF:\n" << current_layer;
      for (size_t k = 1; k < weights.size(); k++) {
        // C = W * C + B
        current_layer = clFTensor::batchedGemm(1.0f, false, weights[k], false, current_layer, 1.0f,
                                               biases[k], queue);
        // std::cout << "current_layer " << k << ":\n" << current_layer;
        layers_output[k + 1].copy(current_layer, queue, false);

        // Apply activation function on every element of the matrix
        af::applyAF(activation_functions[k], current_layer, queue);
        // std::cout << "current_layer AF " << k << ":\n" << current_layer;
        layers_af_output[k + 1].copy(current_layer, queue, false);
      }
    }

    clFTensor backward(MLPerceptron &perceptron, const clFTensor &targets,
                       std::vector<clFTensor> &layers_output,
                       std::vector<clFTensor> &layers_af_output,
                       MLPOptimizer::WeightUpdateCache &updater, cl::CommandQueue &queue) {
      auto &weights = updater.getWeightsCopy();
      auto &activation_functions = perceptron.getActivationFunctions();

      if (weights.empty()) return {};

      // std::stringstream ss;
      // auto substart = std::chrono::high_resolution_clock::now();
      clFTensor error = layers_af_output.back().sub(1.0f, targets, queue);
      // auto subend = std::chrono::high_resolution_clock::now();
      // auto subduration = std::chrono::duration_cast<std::chrono::microseconds>(subend -
      // substart); ss << queue() << ": "
      //    << "Subtraction: " << subduration.count() << " us" << std::endl;
      //   Need to use a long since we stop when index reaches -1
      for (long i = weights.size() - 1; i >= 0; i--) {
        // auto start = std::chrono::high_resolution_clock::now();
        clFTensor derivative;
        derivative.copy(layers_output[i + 1], queue, false);
        // std::cout << "Derivative " << i << ":\n" << derivative;
        af::applyDerivativeAF(activation_functions[i], derivative, queue);
        // std::cout << "Derivative AF " << i << ":\n" << derivative;
        derivative.iphadamard(error, queue);
        // std::cout << "Hadamard " << i << ":\n" << derivative;
        error = clFTensor::batchedGemm(1.0f, true, weights[i], false, derivative, queue);
        // std::cout << "Error " << i << ":\n" << error;

        clFTensor gradient =
                clFTensor::batchedGemm(1.0f, false, derivative, true, layers_af_output[i], queue);
        // std::cout << "Gradient " << i << ":\n" << gradient;

        // auto end = std::chrono::high_resolution_clock::now();
        // auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        // ss << queue() << ": " << i << " layer: Backward pass took " << elapsed.count() << "us"
        //    << std::endl;
        // start = std::chrono::high_resolution_clock::now();
        //  Reduce the gradient to a single matrix
        clFMatrix collapsed_gradient = gradient.sumCollapse(queue);
        // std::cout << "Avg Gradients " << i << ":\n" << derivative;
        //  end = std::chrono::high_resolution_clock::now();
        //  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        //  ss << queue() << ": " << i << " layer: Collapse took " << elapsed.count() << "us"
        //     << std::endl;

        // start = std::chrono::high_resolution_clock::now();
        updater.add(i, collapsed_gradient, gradient.getDepth(), queue);
        // end = std::chrono::high_resolution_clock::now();
        // elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        // ss << queue() << ": " << i << " layer: add took " << elapsed.count() << "us" <<
        // std::endl;
      }
      // std::cout << ss.str();
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

  void WeightUpdateCache::synchronizeWeights(cl::CommandQueue &queue) {
    if (weight_copy.size() != perceptron->getWeights().size()) {
      weight_copy.resize(perceptron->getWeights().size());
    }

    for (size_t i = 0; i < perceptron->getWeights().size(); i++) {
      weight_copy[i].copy(perceptron->getWeights()[i], queue, false);
    }

    if (biases_copy.size() != perceptron->getBiases().size()) {
      biases_copy.resize(perceptron->getBiases().size());
    }

    for (size_t i = 0; i < perceptron->getBiases().size(); i++) {
      biases_copy[i].copy(perceptron->getBiases()[i], queue, false);
    }
  }

  void WeightUpdateCache::acquireBuffer(cl::CommandQueue &queue) {
    size_t buf_count = weight_copy.size() + biases_copy.size() + weight_updates.size();
    std::vector<cl::Memory> buffers;
    buffers.reserve(buf_count);
    for (auto &w : weight_copy) { buffers.push_back(w.getBuffer()); }
    for (auto &b : biases_copy) { buffers.push_back(b.getBuffer()); }
    for (auto &wu : weight_updates) { buffers.push_back(wu.getBuffer()); }
    queue.enqueueMigrateMemObjects(buffers, 0);
  }

  std::unique_ptr<WeightUpdateCache> MLPOptimizer::makeCache(size_t ncache) {
    return std::make_unique<WeightUpdateCache>(*this);
  }

  std::vector<std::unique_ptr<WeightUpdateCache>> MLPOptimizer::makeCaches(size_t ncache) {
    std::vector<std::unique_ptr<WeightUpdateCache>> caches;
    caches.reserve(ncache);
    for (size_t i = 0; i < ncache; i++) { caches.emplace_back(makeCache(ncache)); }
    return caches;
  }

  std::unique_ptr<Optimizer::Operation> MLPOptimizer::makeBatchOperation() {
    return std::make_unique<Operation>(*this);
  }

  clFTensor MLPOptimizer::optimize(const clFTensor &inputs, const clFTensor &targets,
                                   WeightUpdateCache &updater, cl::CommandQueue &queue) {
    std::vector<clFTensor> layers_output(neural_network->getWeights().size() + 1);
    std::vector<clFTensor> layers_af_output(neural_network->getWeights().size() + 1);

    // auto start = std::chrono::high_resolution_clock::now();
    forward(*neural_network, inputs.flatten(), layers_output, layers_af_output, updater, queue);
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // std::stringstream ss;
    // ss << queue() << ": " << inputs.getDepth() << " depth" << std::endl;
    // ss << queue() << ": Forward pass took " << duration.count() << " microseconds" << std::endl;
    // start = std::chrono::high_resolution_clock::now();
    auto res = backward(*neural_network, targets.flatten(), layers_output, layers_af_output,
                        updater, queue);
    // end = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // ss << queue() << ": Backward pass took " << duration.count() << " microseconds" << std::endl;
    // std::cout << ss.str();
    return res;
  }

  math::clFTensor MLPOptimizer::Operation::computeGradient(size_t thread_rank,
                                                           const math::clFTensor &inputs,
                                                           const math::clFTensor &targets,
                                                           cl::CommandQueue &batch_queue) {
    if (thread_rank > caches.size())
      throw std::invalid_argument("Error: Only " + std::to_string(caches.size()) +
                                  " caches reserved, tried to access cache " +
                                  std::to_string(thread_rank));
    caches[thread_rank]->acquireBuffer(batch_queue);
    return optimizer->optimize(inputs, targets, *caches[thread_rank], batch_queue);
  }

  void MLPOptimizer::Operation::reserveCaches(size_t num_threads) {
    if (caches.size() < num_threads) {
      caches = optimizer->makeCaches(num_threads);
      for (auto &cache : caches) { cache->synchronizeWeights(utils::cl_wrapper.getDefaultQueue()); }
      utils::cl_wrapper.getDefaultQueue().finish();
    }
  }

  void MLPOptimizer::Operation::reduceAll(cl::CommandQueue &queue) {
    for (size_t i = 1; i < caches.size(); i++) { caches[0]->reduce(*caches[i], queue); }
  }

  void MLPOptimizer::Operation::applyChanges(cl::CommandQueue &queue) { caches[0]->apply(queue); }

  void MLPOptimizer::Operation::clearChanges(cl::CommandQueue &queue) {
    for (auto &cache : caches) { cache->clear(queue); }
    for (auto &cache : caches) { cache->synchronizeWeights(queue); }
  }
}   // namespace nnet
