#pragma once

#include "MLPModel.hpp"
#include "MLPerceptron.hpp"
#include "Optimization/Optimization.hpp"
#include "Optimizer.hpp"
#include <iostream>
#include <utility>

namespace nnet {
  class MLPOptimizer : public Optimizer {
  public:
    MLPOptimizer(MLPModel &model, std::shared_ptr<Optimization> tm);
    ~MLPOptimizer() override = default;

    MLPOptimizer(const MLPOptimizer &other) = delete;
    MLPOptimizer(MLPOptimizer &&other) noexcept = default;

    virtual void setModel(MLPModel &model) = 0;

    MLPOptimizer &operator=(const MLPOptimizer &other) = delete;
    MLPOptimizer &operator=(MLPOptimizer &&other) noexcept = default;

    MLPerceptron *gePerceptron() const { return neural_network; }
    Optimization *getOptimizationMethod() const { return opti_meth.get(); }

    void update() override { opti_meth->update(); }

  protected:
    MLPerceptron *neural_network;
    std::shared_ptr<Optimization> opti_meth;
  };

  class MLPModelStochOptimizer final : public MLPOptimizer {
  public:
    MLPModelStochOptimizer(MLPModel &model, std::shared_ptr<Optimization> tm);

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


  class MLPBatchOptimizer : public MLPOptimizer {
  public:
    explicit MLPBatchOptimizer(MLPModel &model, std::shared_ptr<Optimization> tm);

    void setModel(MLPModel &model) override;

    void optimize(const std::vector<math::FloatMatrix> &inputs,
                  const std::vector<math::FloatMatrix> &targets) override;

  protected:
    void forward(math::FloatMatrix const &inputs);

    void computeGradient();

    //
    std::vector<math::FloatMatrix> layers, layers_af;

    //
    BackpropStorage storage;

    std::vector<math::FloatMatrix> avg_errors;
    std::vector<math::FloatMatrix> avg_gradients;
  };

  class MLPMiniBatchOptimizer : public MLPBatchOptimizer {
  public:
    explicit MLPMiniBatchOptimizer(MLPModel &model, std::shared_ptr<Optimization> tm,
                                   size_t batch_size = 8);

    void optimize(const std::vector<math::FloatMatrix> &inputs,
                  const std::vector<math::FloatMatrix> &targets) override;

  private:
    size_t batch_size;
  };

}   // namespace nnet