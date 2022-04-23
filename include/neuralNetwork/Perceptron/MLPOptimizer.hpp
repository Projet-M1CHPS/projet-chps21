#pragma once

#include "MLPModel.hpp"
#include "MLPerceptron.hpp"
#include "Optimizer.hpp"
#include "Perceptron/Optimization/Optimization.hpp"
#include "neuralNetwork/OptimizationScheduler/OptimizationScheduler.hpp"
#include <iostream>
#include <utility>

namespace nnet {

  class MLPOptimizer : public Optimizer {
  public:
    class WeightUpdateCache;
    class Operation;
    class WeightUpdater;

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

    virtual std::unique_ptr<Operation> makeMLPOperation();

    math::clFTensor optimize(const math::clFTensor &inputs, const math::clFTensor &targets,
                             WeightUpdateCache &cache, cl::CommandQueue &queue);

    virtual std::unique_ptr<WeightUpdateCache> makeCache();
    virtual std::vector<std::unique_ptr<WeightUpdateCache>> makeCaches(size_t ncache);

  private:
    std::unique_ptr<Optimizer::Operation> makeOperationImpl() override;

    MLPerceptron *neural_network;
    std::unique_ptr<Optimization> opti_meth;
  };

  class MLPOptimizer::WeightUpdateCache {
  public:
    explicit WeightUpdateCache(MLPOptimizer &optimizer);
    WeightUpdateCache(std::vector<math::clFMatrix> weight_updates, size_t contributions);
    WeightUpdateCache(WeightUpdateCache &&other) = default;
    WeightUpdateCache &operator=(WeightUpdateCache &&other) noexcept {
      if (weight_updates.size() != other.weight_updates.size()) {
        tscl::logger("MLPOptimizer::WeightUpdateCache assignment error: different size",
                     tscl::Log::Error);
        tscl::logger("Local size: " + std::to_string(weight_updates.size()), tscl::Log::Error);
        tscl::logger("Other size: " + std::to_string(other.weight_updates.size()),
                     tscl::Log::Error);
        throw std::runtime_error("WeightUpdateCache: incompatible sizes");
      }
      for (size_t i = 0; i < weight_updates.size(); ++i) {
        assert(weight_updates[i].size() == other.weight_updates[i].size());
        assert(weight_updates[i].getRows() == other.weight_updates[i].getRows());
        assert(weight_updates[i].getCols() == other.weight_updates[i].getCols());
        weight_updates[i] = std::move(other.weight_updates[i]);
      }
      contribution = other.getContribution();
      return *this;
    }


    virtual ~WeightUpdateCache() = default;

    math::clFMatrix &operator[](size_t i) { return weight_updates[i]; }

    [[nodiscard]] const math::clFMatrix &operator[](size_t i) const { return weight_updates[i]; }
    [[nodiscard]] const std::vector<math::clFMatrix> &getWeightsCopy() const { return weight_copy; }
    [[nodiscard]] const std::vector<math::clFMatrix> &getBiasesCopy() const { return biases_copy; }
    [[nodiscard]] const std::vector<math::clFMatrix> &getWeightUpdates() const {
      return weight_updates;
    }
    [[nodiscard]] size_t getContribution() const { return contribution; }

    void setContribution(size_t new_contribution) { contribution = new_contribution; }

    void add(size_t index, const math::clFMatrix &delta, size_t contribution_size,
             cl::CommandQueue &queue);
    void reduce(WeightUpdateCache &other, cl::CommandQueue &queue);

    void apply(cl::CommandQueue &queue);

    void increaseContribution(size_t contrib) { contribution += contrib; }

    void synchronizeWeights(cl::CommandQueue &queue);
    void acquireBuffer(cl::CommandQueue &queue);

    void clear(cl::CommandQueue &queue);

  protected:
    MLPerceptron *perceptron{};
    size_t contribution{};
    std::vector<math::clFMatrix> weight_updates;
    std::vector<math::clFMatrix> weight_copy;
    std::vector<math::clFMatrix> biases_copy;

  private:
    Optimization *optimization{};
  };

  class MLPOptimizer::Operation : public Optimizer::Operation {
  public:
    explicit Operation(MLPOptimizer &optimizer) : optimizer(&optimizer) {}

    void operator()(size_t thread_rank, const math::clFTensor &inputs,
                    const math::clFTensor &targets, cl::CommandQueue batch_queue) override {
      computeGradient(thread_rank, inputs, targets, batch_queue);
    }

    math::clFTensor computeGradient(size_t thread_rank, const math::clFTensor &inputs,
                                    const math::clFTensor &targets, cl::CommandQueue batch_queue);

    void reserveCaches(size_t num_threads) override;

    WeightUpdateCache &getCache(size_t thread_rank) {
      if (thread_rank >= caches.size()) {
        throw std::runtime_error("Cache not allocated for thread " + std::to_string(thread_rank));
      }
      return *caches[thread_rank];
    }


  protected:
    std::vector<std::unique_ptr<WeightUpdateCache>> caches;
    MLPOptimizer *optimizer;

    virtual void reduceAll(cl::CommandQueue &queue);
    virtual void applyChanges(cl::CommandQueue &queue);
    virtual void clearChanges(cl::CommandQueue &queue);
  };

}   // namespace nnet