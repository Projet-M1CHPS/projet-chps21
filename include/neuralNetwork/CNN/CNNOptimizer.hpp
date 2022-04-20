#pragma once

#include "CNNModel.hpp"
#include "Optimizer.hpp"
#include "Perceptron/MLPModel.hpp"
#include "Perceptron/MLPOptimizer.hpp"
#include "Perceptron/Optimization/Optimization.hpp"
#include "neuralNetwork/CNN/CNN.hpp"
#include "neuralNetwork/CNN/CNNStorageBP.hpp"
#include <iostream>
#include <utility>

// TODO : implement me correctly
class CNNOptimization {
public:
  virtual void update(math::clFTensor &weight, math::clFTensor &gradient, cl::CommandQueue &queue) {

  }
};

namespace nnet {

  class CNNOptimizer : public Optimizer {
  public:
    class WeightUpdateCache;
    class Operation;

    CNNOptimizer(CNNModel &model, std::unique_ptr<Optimization> mlp_optimization);

    CNNOptimizer(const CNNOptimizer &other) = delete;
    CNNOptimizer(CNNOptimizer &&other) noexcept = default;

    CNNOptimizer &operator=(const CNNOptimizer &other) = delete;
    CNNOptimizer &operator=(CNNOptimizer &&other) noexcept = default;

    void update() override { mlp_optimizer.update(); }

    void optimize(const math::clFTensor &inputs, const math::clFTensor &targets,
                  WeightUpdateCache &cnn_cache, MLPOptimizer::WeightUpdateCache &mlp_cache,
                  cl::CommandQueue &queue);

    std::unique_ptr<Operation> makeCNNOperation();

    std::unique_ptr<WeightUpdateCache> makeCache();
    std::vector<std::unique_ptr<WeightUpdateCache>> makeCaches(size_t ncache);

  private:
    std::unique_ptr<Optimizer::Operation> makeOperationImpl() override;

    CNN *cnn;
    MLPOptimizer mlp_optimizer;
    std::unique_ptr<CNNOptimization> optimization;
  };

  class CNNOptimizer::WeightUpdateCache {
  public:
    explicit WeightUpdateCache(CNNOptimizer &optimizer);
    virtual ~WeightUpdateCache() = default;

    WeightUpdateCache(WeightUpdateCache &&other) noexcept = default;

    WeightUpdateCache(CNN *cnn, std::vector<math::clFTensor> &&weight_updates, size_t contribution);

    void add(size_t index, const math::clFTensor &delta, cl::CommandQueue &queue);

    void reduce(WeightUpdateCache &other, cl::CommandQueue &queue);

    void apply(cl::CommandQueue &queue);

    void increaseContribution(size_t contribution) { contributions += contribution; }

    std::vector<std::unique_ptr<CNNLayer>> &getLayers() { return layers_copy; }

    void synchronizeLayers(cl::CommandQueue &queue);

    void clear(cl::CommandQueue &queue);

  protected:
    CNN *cnn;
    size_t contributions;
    std::vector<math::clFTensor> weight_updates;
    std::vector<std::unique_ptr<CNNLayer>> layers_copy;

  private:
    CNNOptimization *optimization;
  };


  class CNNOptimizer::Operation : public Optimizer::Operation {
  public:
    explicit Operation(CNNOptimizer &optimizer)
        : mlp_operation(optimizer.mlp_optimizer.makeMLPOperation()), optimizer(&optimizer) {}

    void operator()(size_t thread_rank, const math::clFTensor &inputs,
                    const math::clFTensor &targets, cl::CommandQueue batch_queue) override {
      optimizer->optimize(inputs, targets, *caches[thread_rank],
                          mlp_operation->getCache(thread_rank), batch_queue);
    }

    void reserveCaches(size_t num_threads) override {
      if (caches.size() < num_threads) {
        caches = optimizer->makeCaches(num_threads);
        for (auto &cache : caches) cache->synchronizeLayers(utils::cl_wrapper.getDefaultQueue());
        utils::cl_wrapper.getDefaultQueue().finish();
      }
      mlp_operation->reserveCaches(num_threads);
    }

  protected:
    std::vector<std::unique_ptr<WeightUpdateCache>> caches;
    std::unique_ptr<MLPOptimizer::Operation> mlp_operation;
    CNNOptimizer *optimizer;

  private:
    void reduceAll(cl::CommandQueue &queue) override {
      for (auto &cache : caches) { cache->reduce(*caches[0], queue); }
    }

    void applyChanges(cl::CommandQueue &queue) override {
      caches[0]->apply(queue);
      mlp_operation->updateModel(queue);
    }

    void clearChanges(cl::CommandQueue &queue) override {
      for (auto &cache : caches) { cache->clear(queue); }
      for (auto &cache : caches) { cache->synchronizeLayers(queue); }
    }
  };


}   // namespace nnet