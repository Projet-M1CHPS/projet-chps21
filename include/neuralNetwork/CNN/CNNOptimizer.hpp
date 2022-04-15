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
    CNNOptimizer(CNNModel &model);

    CNNOptimizer(const CNNOptimizer &other) = delete;
    CNNOptimizer(CNNOptimizer &&other) noexcept = default;

    CNNOptimizer &operator=(const CNNOptimizer &other) = delete;
    CNNOptimizer &operator=(CNNOptimizer &&other) noexcept = default;

    CNN *getCNN() const { return cnn; }
    MLPerceptron *getMLP() const { return mlp; }

    // TODO : attention computeGradient doit etre override
    void optimize(const clFTensor &inputs, const clFTensor &targets);

    // TODO : implémenter update
    void update() override {}

  private:
    clFTensor forward(const clFTensor &inputs,
                      std::vector<std::unique_ptr<CNNStorageBP>> &storages);
    void backward(const clFTensor &inputs, const clFTensor &errorsFlatten,
                  std::vector<std::unique_ptr<CNNStorageBP>> &storages);

  private:
    CNN *cnn;
    MLPerceptron *mlp;
  };

}   // namespace nnet