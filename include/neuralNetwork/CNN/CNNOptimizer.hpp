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


namespace nnet {
  using namespace nnet;
  using namespace math;

  class CNNOptimizer : public Optimizer {
  public:
    class CNNWeightUpdater;
    class CNNOperation;

    CNNOptimizer(CNNModel &model, std::unique_ptr<Optimization> mlpTm);

    CNNOptimizer(const CNNOptimizer &other) = delete;
    CNNOptimizer(CNNOptimizer &&other) noexcept = default;

    CNNOptimizer &operator=(const CNNOptimizer &other) = delete;
    CNNOptimizer &operator=(CNNOptimizer &&other) noexcept = default;

    // TODO : implémenter update
    void update() override {}


    /*template<class optim, typename... Args>
    static std::unique_ptr<CNNOptimizer> make(CNNModel &model, Args &&...args) {
      return std::make_unique<CNNOptimizer>(
              model, std::make_unique<optim>(model.getPerceptron(), std::forward<Args>(args)...));
    }*/

    void optimize(const clFTensor &inputs, const clFTensor &targets,
                  MLPOptimizer::WeightUpdater &mlpIpdater, cl::CommandQueue &queue);

    std::unique_ptr<Optimizer::Operation> makeBatchOperation() override;

  private:
    CNN *cnn;
    MLPOptimizer mlpOptimizer;
  };

  class CNNOptimizer::CNNWeightUpdater : public MLPOptimizer::WeightUpdater {
  public:
    CNNWeightUpdater(MLPerceptron &parent, Optimization &opt);
    virtual ~CNNWeightUpdater() = default;


    void reduce(size_t index, const math::clFTensor &delta, size_t contribution_size,
                cl::Event &event);

    virtual void apply();

  protected:
    MLPerceptron *perceptron;
    std::vector<clFMatrix> weight_updates;

  private:
    //CNNOptimization *optimization;

    cl::CommandQueue work_queue;

    std::mutex mutex;
    std::vector<size_t> contributions;
  };

  class CNNOptimizer::CNNOperation : public Optimizer::Operation {
  public:
    CNNOperation(CNN *cnn, MLPerceptron *mlp, MLPOptimizer &mlpOptimizer,
              std::unique_ptr<MLPOptimizer::WeightUpdater> mlpWeightUpdater);

    virtual ~CNNOperation() = default;

    void operator()(const clFTensor &inputs, const clFTensor &targets,
                    cl::Device &device) override {
      cl::CommandQueue queue(utils::cl_wrapper.getContext(), device);
      optimizer->optimize(inputs, targets, *mlpWeightUpdater, queue);
    }

  private:
    std::unique_ptr<MLPOptimizer::WeightUpdater> mlpWeightUpdater;
    CNNOptimizer *optimizer;
  };

}   // namespace nnet