#include "neuralNetwork/CNN/CNNOptimizer.hpp"


namespace nnet {
  namespace {
    math::clFTensor forward(const CNN &cnn, const math::clFTensor &inputs,
                      std::vector<std::unique_ptr<CNNStorageBP>> &storages,
                      cl::CommandQueue &queue) {
      auto &layers = cnn.getLayers();

      math::clFTensor output = inputs.shallowCopy();

      for (size_t i = 0; i < layers.size(); i++) {
        output = layers[i]->computeForward(output, *storages[i]);
      }

      reorganizeForward(queue, output, inputs.getDepth(), cnn.getTopology().getNBranchFinal());

      return output;
    }

    void backward(CNNOptimizer::WeightUpdateCache &cache, const CNN &cnn, const math::clFTensor &inputs,
                  math::clFTensor &errorsFlatten, std::vector<std::unique_ptr<CNNStorageBP>> &storages,
                  cl::CommandQueue &queue) {
      auto &layers = cache.getLayers();

      reorganizeBackward(queue, errorsFlatten, errorsFlatten.getDepth(),
                         cnn.getTopology().getNBranchFinal(), layers.back()->getOutputSize());

      math::clFTensor output = inputs.shallowCopy();

      for (long i = static_cast<long>(layers.size() - 1); i > -1; i--) {
        output = layers[i]->computeBackward(output, *storages[i]);
      }

      size_t i = 0;
      for (auto &s : storages) {
        if (s->hasGradient()) {
          cache.add(i, s->getGradient(), queue);
          i++;
        }
      }
    }
  }   // namespace

  using WeightUpdateCache = CNNOptimizer::WeightUpdateCache;

  WeightUpdateCache::WeightUpdateCache(CNNOptimizer &optimizer)
      : cnn(optimizer.cnn), contributions(0), optimization(optimizer.optimization.get()) {
    auto &layers = cnn->getLayers();

    for (auto &l : layers) {
      if (l->hasWeight()) {
        const auto &filter = l->getWeight();
        weight_updates.emplace_back(filter.getRows(), filter.getCols(), filter.getDepth());
      }
    }
  }

  WeightUpdateCache::WeightUpdateCache(CNN *cnn, std::vector<math::clFTensor> &&weight_updates,
                                       size_t contribution) {}

  void WeightUpdateCache::add(size_t index, const math::clFTensor &delta, cl::CommandQueue &queue) {
    weight_updates[index].ipadd(1.0f, delta, queue);
  }

  void WeightUpdateCache::reduce(WeightUpdateCache &other, cl::CommandQueue &queue) {
    for (size_t i = 0; i < weight_updates.size(); ++i) {
      weight_updates[i].ipadd(1.0f, other.weight_updates[i], queue);
    }
  }

  void WeightUpdateCache::apply(cl::CommandQueue &queue) {
    size_t tensor_index = 0;
    auto &cnn_layers = cnn->getLayers();

    for (auto &layer : cnn_layers) {
      if (layer->hasWeight()) {
        auto &filter = layer->getWeight();
        optimization->update(filter, weight_updates[tensor_index], queue);
        tensor_index++;
      }
    }
  }

  void WeightUpdateCache::synchronizeLayers(cl::CommandQueue &queue) {
    layers_copy = cnn->copyLayers();
  }

  void WeightUpdateCache::clear(cl::CommandQueue &queue) {
    for (auto &tensor : weight_updates) { tensor.fill(0.f, queue, false); }
  }

  CNNOptimizer::CNNOptimizer(CNNModel &model, std::unique_ptr<Optimization> mlpTm)
      : cnn(&model.getCnn()), mlp_optimizer(model.getMlp(), std::move(mlpTm)) {}

  void CNNOptimizer::optimize(const math::clFTensor &inputs, const math::clFTensor &targets,
                              WeightUpdateCache &cnn_cache,
                              MLPOptimizer::WeightUpdateCache &mlp_cache, cl::CommandQueue &queue) {
    std::vector<std::unique_ptr<CNNStorageBP>> storages = cnn->getTopology().convertToStorage();

    math::clFTensor flatten = forward(*cnn, inputs, storages, queue);

    math::clFTensor errorFlatten = mlp_optimizer.optimize(flatten, targets, mlp_cache, queue);

    backward(cnn_cache, *cnn, inputs, errorFlatten, storages, queue);
    cnn_cache.increaseContribution(inputs.getDepth());
  }

  std::unique_ptr<WeightUpdateCache> CNNOptimizer::makeCache() {
    return std::make_unique<WeightUpdateCache>(*this);
  }

  std::vector<std::unique_ptr<WeightUpdateCache>> CNNOptimizer::makeCaches(size_t ncache) {
    std::vector<std::unique_ptr<WeightUpdateCache>> caches;
    caches.reserve(ncache);
    for (size_t i = 0; i < ncache; ++i) { caches.emplace_back(makeCache()); }
    return caches;
  }

  std::unique_ptr<CNNOptimizer::Operation> CNNOptimizer::makeCNNOperation() {
    return std::make_unique<Operation>(*this);
  }


  std::unique_ptr<Optimizer::Operation> CNNOptimizer::makeOperationImpl() {
    return makeCNNOperation();
  }

}   // namespace nnet