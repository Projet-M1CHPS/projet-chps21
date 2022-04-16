#pragma once

#include "MLPModel.hpp"
#include "MLPerceptron.hpp"
#include "Optimization.hpp"
#include "Optimizer.hpp"
#include "neuralNetwork/OptimizationScheduler/OptimizationScheduler.hpp"
#include <iostream>
#include <utility>

namespace nnet {

  class MLPOptimizer : public Optimizer {
  public:
    class WeightUpdateCache;
    class Operation;

    MLPOptimizer(MLPModel &model, std::unique_ptr<Optimization> tm)
        : neural_network(&model.getPerceptron()), opti_meth(std::move(tm)) {}

    MLPOptimizer(MLPerceptron &perceptron, std::unique_ptr<Optimization> tm)
        : neural_network(&perceptron), opti_meth(std::move(tm)) {}

    void update() override { opti_meth->update(); }

    template<class optim, typename... Args>
    static std::unique_ptr<MLPOptimizer> make(MLPModel &model, Args &&...args) {
      return std::make_unique<MLPOptimizer>(
              model, std::make_unique<optim>(model.getPerceptron(), std::forward<Args>(args)...));
    }

    math::clFTensor optimize(const math::clFTensor &inputs, const math::clFTensor &targets,
                             WeightUpdateCache &updater, cl::CommandQueue &queue);

    std::unique_ptr<Optimizer::Operation> makeBatchOperation() override;

    virtual std::unique_ptr<WeightUpdateCache> makeCache(size_t ncache);
    virtual std::vector<std::unique_ptr<WeightUpdateCache>> makeCaches(size_t ncache);

  private:
    MLPerceptron *neural_network;
    std::unique_ptr<Optimization> opti_meth;
  };

  class MLPOptimizer::WeightUpdateCache {
  public:
    explicit WeightUpdateCache(MLPOptimizer &optimizer);
    virtual ~WeightUpdateCache() = default;

    math::clFMatrix &operator[](size_t i) { return weight_updates[i]; }

    const math::clFMatrix &operator[](size_t i) const { return weight_updates[i]; }

    void add(size_t index, const math::clFMatrix &delta, size_t contribution_size,
             cl::CommandQueue &queue);
    void reduce(WeightUpdateCache &other, cl::CommandQueue &queue);

    void apply(cl::CommandQueue &queue);

    void clear(cl::CommandQueue& queue);

  protected:
    MLPerceptron *perceptron;
    std::vector<size_t> contributions;
    std::vector<math::clFMatrix> weight_updates;

  private:
    Optimization *optimization;
  };

  class MLPOptimizer::Operation : public Optimizer::Operation {
  public:
    explicit Operation(MLPOptimizer &optimizer) : optimizer(&optimizer) {}

    ~Operation() override = default;

    void operator()(size_t thread_rank, const math::clFTensor &inputs,
                    const math::clFTensor &targets, cl::CommandQueue &batch_queue) override {
      computeGradient(thread_rank, inputs, targets, batch_queue);
    }

    math::clFTensor computeGradient(size_t thread_rank, const math::clFTensor &inputs,
                                    const math::clFTensor &targets, cl::CommandQueue &batch_queue);

    void reserveCaches(size_t num_threads) override;

  protected:
    std::vector<std::unique_ptr<WeightUpdateCache>> caches;
    MLPOptimizer *optimizer;

    void reduceAll(cl::CommandQueue &queue) override;
    void applyChanges(cl::CommandQueue &queue) override;
    void clearChanges(cl::CommandQueue &queue) override;
  };

}   // namespace nnet