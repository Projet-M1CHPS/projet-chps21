#pragma once

#include "CNNModel.hpp"
#include "Optimizer.hpp"
#include "Perceptron/MLPModel.hpp"
#include "Perceptron/MLPOptimizer.hpp"
#include "Perceptron/MLPStochOptimizer.hpp"
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
    CNNOptimizer(CNNModel &model);

    CNNOptimizer(const CNNOptimizer &other) = delete;
    CNNOptimizer(CNNOptimizer &&other) noexcept = default;

    CNNOptimizer &operator=(const CNNOptimizer &other) = delete;
    CNNOptimizer &operator=(CNNOptimizer &&other) noexcept = default;

    CNN *getCNN() const { return nn_cnn; }
    MLPerceptron *getMLP() const { return nn_mlp; }

  public:
    CNN *nn_cnn;
    MLPerceptron *nn_mlp;
  };

  class CNNStochOptimizer final : public CNNOptimizer {
  public:
    CNNStochOptimizer(CNNModel &model, std::unique_ptr<Optimization> mlp_optimization)
        : CNNOptimizer(model), mlp_opti(model.getMlp(), std::move(mlp_optimization)) {}

    void optimize(const math::clFMatrix &input, const math::clFMatrix &target);

    void optimize(const std::vector<math::clFTensor> &inputs,
                  const std::vector<math::clFTensor> &targets) override;

    void update() override { mlp_opti.update(); }

    template<typename Optimization, typename... Args,
             typename = std::is_base_of<nnet::Optimization, Optimization>>
    static std::unique_ptr<CNNStochOptimizer> make(CNNModel &model, Args &&...args) {
      return std::make_unique<CNNStochOptimizer>(
              model, std::make_unique<Optimization>(model.getMlp(), std::forward<Args>(args)...));
    }

  private:
    clFMatrix forward(const clFMatrix &input);
    void backward(const clFMatrix &target, const clFMatrix &errorFlatten);

    MLPStochOptimizer mlp_opti;
  };


}   // namespace nnet