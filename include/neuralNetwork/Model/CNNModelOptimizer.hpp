#pragma once

#include "CNNModel.hpp"
#include "MLPModel.hpp"
#include "MLPModelOptimizer.hpp"
#include "ModelOptimizer.hpp"
#include "neuralNetwork/CNN/CNN.hpp"
#include "neuralNetwork/CNN/CNNStorageBP.hpp"
#include "neuralNetwork/Perceptron/OptimizationMethod.hpp"
#include <iostream>
#include <utility>


namespace cnnet {
  using namespace nnet;
  using namespace math;

  class CNNModelOptimizer : public ModelOptimizer {
  public:
    CNNModelOptimizer(CNNModel &model, std::shared_ptr<OptimizationMethod> tm);
    virtual ~CNNModelOptimizer() override = default;

    CNNModelOptimizer(const CNNModelOptimizer &other) = delete;
    CNNModelOptimizer(CNNModelOptimizer &&other) noexcept = default;

    virtual void setModel(MLPModel &model) {}

    CNNModelOptimizer &operator=(const CNNModelOptimizer &other) = delete;
    CNNModelOptimizer &operator=(CNNModelOptimizer &&other) noexcept = default;

    CNN *getCnn() const { return nn_cnn; }
    // MLPerceptron *getMlp() const { return mlpOpti.getPerceptron(); }
    OptimizationMethod *getOptimizationMethod() const { return opti_meth.get(); }

    void update() override { opti_meth->update(); }

    virtual void train(const math::FloatMatrix &input, const math::FloatMatrix &target) = 0;

    virtual void optimize(const std::vector<math::FloatMatrix> &inputs,
                          const std::vector<math::FloatMatrix> &targets) = 0;

  public:
    virtual void forward(math::FloatMatrix const &input) = 0;
    virtual void backward(math::FloatMatrix const &input,
                          math::FloatMatrix const &errorFlatten) = 0;

  public:
    CNN *nn_cnn;
    FloatMatrix *flatten;
    MLPerceptron *nn_mlp;

    std::vector<std::vector<std::shared_ptr<CNNStorageBP>>> storage;

    std::shared_ptr<OptimizationMethod> opti_meth;
  };


  class CNNModelStochOptimizer final : public CNNModelOptimizer {
  public:
    CNNModelStochOptimizer(CNNModel &model, std::shared_ptr<OptimizationMethod> tm);

    void train(const math::FloatMatrix &input, const math::FloatMatrix &target) override;

    void optimize(const std::vector<math::FloatMatrix> &inputs,
                  const std::vector<math::FloatMatrix> &targets) override;

  public:
    void forward(math::FloatMatrix const &input);

    void backward(math::FloatMatrix const &target, math::FloatMatrix const &errorFlatten);

  private:
    MLPModelStochOptimizer mlpOpti;
  };


}   // namespace cnnet