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


namespace cnnet {
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

    FloatMatrix flatten;

    // TODO: Move inside the layer ?
    std::vector<std::vector<std::shared_ptr<CNNStorageBP>>> storage;
  };

  class CNNStochOptimizer final : public CNNOptimizer {
  public:
    CNNStochOptimizer(CNNModel &model, std::unique_ptr<Optimization> mlp_optimization)
        : CNNOptimizer(model),
          mlp_opti(model.getMlp(), &model.getClWrapper(), std::move(mlp_optimization)) {}

    void optimize(const math::clFMatrix &input, const math::clFMatrix &target);

    void optimize(const std::vector<math::clFMatrix> &inputs,
                  const std::vector<math::clFMatrix> &targets) override;

    void update() override { mlp_opti.update(); }

    template<typename Optimization, typename... Args,
             typename = std::is_base_of<nnet::Optimization, Optimization>>
    std::unique_ptr<CNNStochOptimizer> make(CNNModel &model, Args &&...args) {
      return std::make_unique<CNNStochOptimizer>(
              model, std::make_unique<Optimization>(model.getMlp(), model.getClWrapper(),
                                                    std::forward<Args>(args)...));
    }

  private:
    // TODO : Use clFMatrix
    void forward(math::FloatMatrix const &input);

    // TODO : Use clFMatrix
    void backward(math::FloatMatrix const &target, math::FloatMatrix const &errorFlatten);

    MLPStochOptimizer mlp_opti;
  };


}   // namespace cnnet