#pragma once

#include "MLPModel.hpp"
#include "ModelOptimizer.hpp"
#include "neuralNetwork/Perceptron/MLPerceptron.hpp"
#include "neuralNetwork/Perceptron/OptimizationMethod.hpp"
#include <iostream>
#include <utility>

namespace nnet {
  class MLPModelOptimizer : public ModelOptimizer {
  public:
    MLPModelOptimizer(MLPModel &model, std::shared_ptr<OptimizationMethod> tm);
    ~MLPModelOptimizer() override = default;

    MLPModelOptimizer(const MLPModelOptimizer &other) = delete;
    MLPModelOptimizer(MLPModelOptimizer &&other) noexcept = default;

    virtual void setModel(MLPModel &model) = 0;

    MLPModelOptimizer &operator=(const MLPModelOptimizer &other) = delete;
    MLPModelOptimizer &operator=(MLPModelOptimizer &&other) noexcept = default;

    MLPerceptron *gePerceptron() const { return neural_network; }
    OptimizationMethod *getOptimizationMethod() const { return opti_meth.get(); }

    void update() override { opti_meth->update(); }

  protected:
    MLPerceptron *neural_network;
    std::shared_ptr<OptimizationMethod> opti_meth;
  };

  class MLPModelStochOptimizer final : public MLPModelOptimizer {
  public:
    MLPModelStochOptimizer(MLPModel &model, std::shared_ptr<OptimizationMethod> tm);

    void train(const math::FloatMatrix &input, const math::FloatMatrix &target);

    void setModel(MLPModel &model) override;

    void optimize(const std::vector<math::FloatMatrix> &inputs,
                  const std::vector<math::FloatMatrix> &targets) override;

  private:
    void forward(math::FloatMatrix const &inputs);

    void backward(math::FloatMatrix const &target);

  private:
    //
    std::vector<math::FloatMatrix> layers, layers_af;

    //
    BackpropStorage storage;
  };


  class MLPBatchOptimizer : public MLPModelOptimizer {
  public:
    explicit MLPBatchOptimizer(MLPModel &model, std::shared_ptr<OptimizationMethod> tm);

    void setModel(MLPModel &model) override;

    void optimize(const std::vector<math::FloatMatrix> &inputs,
                  const std::vector<math::FloatMatrix> &targets) override;

  protected:
    void forward(math::FloatMatrix const &inputs);

    void computeGradient();

    void backward(math::FloatMatrix const &target);

    //
    std::vector<math::FloatMatrix> layers, layers_af;

    //
    BackpropStorage storage;

    std::vector<math::FloatMatrix> avg_errors;
    std::vector<math::FloatMatrix> avg_gradients;
  };

  class MLPMiniBatchOptimizer : public MLPBatchOptimizer {
  public:
    explicit MLPMiniBatchOptimizer(MLPModel &model,
                                   std::shared_ptr<OptimizationMethod> tm,
                                   size_t batch_size = 8);

    void optimize(const std::vector<math::FloatMatrix> &inputs,
                  const std::vector<math::FloatMatrix> &targets) override;

  private:
    size_t batch_size;
  };

}   // namespace nnet