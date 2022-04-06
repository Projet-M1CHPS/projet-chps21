#pragma once

#include "MLPModel.hpp"
#include "MLPerceptron.hpp"
#include "Optimization/RProPOptimization.hpp"
#include "Optimizer.hpp"
#include <iostream>
#include <utility>

namespace nnet {
  class MLPOptimizer : public Optimizer {
  public:
    MLPOptimizer(MLPModel &model, std::unique_ptr<Optimization> tm)
        : neural_network(&model.getPerceptron()), opti_meth(std::move(tm)) {}

    MLPOptimizer(MLPerceptron &perceptron, std::unique_ptr<Optimization> tm)
        : neural_network(&perceptron), opti_meth(std::move(tm)) {}

    /**
     * @brief Train the neural network on a single input and return the error on the input
     * Mainly used for convolution network and testing
     * @param input
     * @param target
     * @return
     */
    void optimize(const std::vector<math::clFTensor> &inputs,
                  const std::vector<math::clFTensor> &targets) override = 0;

    void update() override { opti_meth->update(); }

  protected:
    MLPerceptron *neural_network;
    std::unique_ptr<Optimization> opti_meth;
  };
}   // namespace nnet