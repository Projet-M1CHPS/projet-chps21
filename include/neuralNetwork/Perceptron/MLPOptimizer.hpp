#pragma once

#include "BatchScheduler.hpp"
#include "MLPModel.hpp"
#include "MLPerceptron.hpp"
#include "Optimization.hpp"
#include "Optimizer.hpp"
#include <iostream>
#include <utility>

namespace nnet {

  class MLPWeightUpdater {
  public:
    MLPWeightUpdater(MLPerceptron &parent, Optimization &optimization);
    const math::clFMatrix &operator[](size_t i);

    void reduce(size_t index, const math::clFMatrix &delta, cl::Event &event);
    virtual void apply();

  private:
    std::mutex mutex;
    MLPerceptron *perceptron;
    Optimization *optimization;

    cl::CommandQueue work_queue;

    size_t n_count;
    std::vector<math::clFMatrix> weight_updates;
  };

  class MLPOptimizer : public Optimizer {
  public:
    MLPOptimizer(MLPModel &model, std::unique_ptr<Optimization> tm, size_t batch_size)
        : neural_network(&model.getPerceptron()), opti_meth(std::move(tm)) {}

    MLPOptimizer(MLPerceptron &perceptron, std::unique_ptr<Optimization> tm)
        : neural_network(&perceptron), opti_meth(std::move(tm)) {}

    void update() override { opti_meth->update(); }

    size_t getBatchSize() const { return batch_size; }
    size_t setBatchSize() const { return batch_size; }

    template<class optim, typename... Args>
    static std::unique_ptr<MLPOptimizer> make(MLPModel &model, size_t batch_size, Args &&...args) {
      return std::make_unique<MLPOptimizer>(
              model, std::make_unique<optim>(model.getPerceptron(), std::forward<Args>(args)...),
              batch_size);
    }

    /**
     * @brief Train the neural network on a single input and return the error on the input
     * Mainly used for convolution network and testing
     * @param input
     * @param target
     * @return
     */
    void optimize(const std::vector<math::clFTensor> &inputs,
                  const std::vector<math::clFTensor> &targets) override;

    void optimize(const math::clFTensor &inputs, const math::clFTensor &targets,
                  MLPWeightUpdater &updater, cl::CommandQueue &queue);

  private:
    MLPerceptron *neural_network;
    std::unique_ptr<Optimization> opti_meth;
    size_t batch_size;
  };

  class MLPBatchScheduler : public BatchScheduler {
  public:
    MLPBatchScheduler(MLPOptimizer &optimizer, MLPWeightUpdater &updater, size_t batch_size)
        : BatchScheduler(batch_size), optimizer(&optimizer), updater(&updater) {}

    void endOfBatch() override { updater->apply(); }

    void runBatch(const math::clFTensor &inputs, const math::clFTensor &targets,
                  cl::CommandQueue &queue) override {
      optimizer->optimize(inputs, targets, *updater, queue);
    }

  private:
    MLPWeightUpdater *updater;
    MLPOptimizer *optimizer;
  };

}   // namespace nnet