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
    MLPOptimizer(MLPModel &model, std::shared_ptr<Optimization> tm)
        : neural_network(&model.getPerceptron()), opti_meth(std::move(tm)) {}

    MLPerceptron *gePerceptron() const { return neural_network; }
    Optimization *getOptimizationMethod() const { return opti_meth.get(); }

    /**
     * @brief Train the neural network on a single input and return the error on the input
     * Mainly used for convolution network and testing
     * @param input
     * @param target
     * @return
     */
    virtual void optimize(const std::vector<math::FloatMatrix> &inputs,
                          const std::vector<math::FloatMatrix> &targets);


    void update() override { opti_meth->update(); }

  protected:
    MLPerceptron *neural_network;
    std::shared_ptr<Optimization> opti_meth;
  };
}   // namespace nnet