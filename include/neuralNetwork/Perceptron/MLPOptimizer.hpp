#pragma once

#include "MLPModel.hpp"
#include "MLPerceptron.hpp"
#include "Optimization.hpp"
#include "neuralNetwork/OptimizationScheduler/OptimizationScheduler.hpp"
#include "Optimizer.hpp"
#include <iostream>
#include <utility>

namespace nnet {

  class MLPOptimizer : public Optimizer {
  public:
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

    math::clFTensor optimize(const math::clFTensor &inputs, const math::clFTensor &targets,
                             WeightUpdater &updater, cl::CommandQueue &queue);

    std::unique_ptr<Optimizer::Operation> makeBatchOperation() override;

  private:

    MLPerceptron *neural_network;
    std::unique_ptr<Optimization> opti_meth;
  };

  class MLPOptimizer::WeightUpdater {
  public:
    WeightUpdater(MLPerceptron &parent, Optimization &opt);
    virtual ~WeightUpdater() = default;

    const math::clFMatrix &operator[](size_t i);

    void reduce(size_t index, const math::clFMatrix &delta, size_t contribution_size,
                cl::Event &event);
    virtual void apply();

  protected:
    MLPerceptron *perceptron;

  private:
    Optimization *optimization;

    cl::CommandQueue work_queue;

    std::mutex mutex;
    std::vector<size_t> contributions;
    std::vector<math::clFMatrix> weight_updates;
  };

  class MLPOptimizer::Operation : public Optimizer::Operation {
  public:
    Operation(MLPOptimizer &optimizer, std::shared_ptr<WeightUpdater> updater)
        : updater(std::move(updater)), optimizer(&optimizer) {}

    ~Operation() override = default;

    void operator()(const math::clFTensor &inputs, const math::clFTensor &targets,
                    cl::CommandQueue &batch_queue) override {
      optimizer->optimize(inputs, targets, *updater, batch_queue);
    }

    void updateModel() override { updater->apply(); }

  protected:
    std::shared_ptr<WeightUpdater> updater;
    MLPOptimizer *optimizer;
  };

}   // namespace nnet