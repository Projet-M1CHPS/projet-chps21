#pragma once

#include "MLPModel.hpp"
#include "MLPerceptron.hpp"
#include "Optimization.hpp"
#include "Optimizer.hpp"
#include "OptimizerScheduler.hpp"
#include <iostream>
#include <utility>

namespace nnet {

  class MLPWeightUpdater {
  public:
    MLPWeightUpdater(MLPerceptron &parent, Optimization &opt);
    const math::clFMatrix &operator[](size_t i);

    void reduce(size_t index, const math::clFMatrix &delta, size_t contribution_size, cl::Event &event);
    virtual void apply();

  private:
    MLPerceptron *perceptron;
    Optimization *optimization;

    cl::CommandQueue work_queue;

    std::vector<std::atomic<size_t>> contributions;
    std::vector<math::clFMatrix> weight_updates;
  };

  class MLPOptimizer : public Optimizer {
  public:
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

    void optimize(const math::clFTensor &inputs, const math::clFTensor &targets,
                  MLPWeightUpdater &updater, cl::CommandQueue &queue);

    std::unique_ptr<OptimizerOperation> makeBatchOperation() override;

  private:
    MLPerceptron *neural_network;
    std::unique_ptr<Optimization> opti_meth;
  };

  class MLPBatchOperation : public OptimizerOperation {
  public:
    MLPBatchOperation(MLPOptimizer &optimizer, std::shared_ptr<MLPWeightUpdater> updater)
        : optimizer(&optimizer), updater(std::move(updater)) {}

    void operator()(const math::clFTensor &inputs, const math::clFTensor &targets,
                    cl::Device &batch_device) override {
      cl::CommandQueue queue(utils::cl_wrapper.getContext(), batch_device);
      optimizer->optimize(inputs, targets, *updater, queue);
    }

    void updateModel() override { updater->apply(); }

  protected:
    std::shared_ptr<MLPWeightUpdater> updater;
    MLPOptimizer *optimizer;
  };

}   // namespace nnet