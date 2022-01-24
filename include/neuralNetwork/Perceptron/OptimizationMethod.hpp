#pragma once

#include <vector>

#include "BackpropStorage.hpp"
#include "MLPerceptron.hpp"
#include "Matrix.hpp"


namespace nnet {


  /** Keep this in sync with new classes
   *
   */
  enum class OptimizationAlgorithm { standard, decay, momentum, rpropPlus };

  class OptimizationMethod {
  public:
    OptimizationMethod() = default;
    virtual ~OptimizationMethod() = default;

    virtual void compute(BackpropStorage &storage) = 0;
    virtual void update(){};
  };

  class SGDOptimization : public OptimizationMethod {
  public:
    SGDOptimization(const float learningRate) : learning_r(learningRate) {}

    void compute(BackpropStorage &storage) override;

  private:
    const float learning_r;
  };

  class DecayOptimization : public OptimizationMethod {
  public:
    DecayOptimization(const float lr_0, const float dr)
        : initial_lr(lr_0), decay_r(dr), curr_lr(lr_0), epoch(0) {}

    void compute(BackpropStorage &storage) override;
    void update() override;

  private:
    const float initial_lr;
    const float decay_r;
    float curr_lr;

    size_t epoch = 0;
  };

  class MomentumOptimization : public OptimizationMethod {
  public:
    MomentumOptimization(MLPerceptron &perceptron, const float learning_rate,
                         const float momentum)
        : lr(learning_rate), momentum(momentum) {
      setPerceptron(perceptron);
    }

    void setPerceptron(MLPerceptron &perceptron);
    void compute(BackpropStorage &storage);

  private:
    const float lr;
    const float momentum;
    std::vector<math::FloatMatrix> old_weight_change;
  };

  class DecayMomentumOptimization : public OptimizationMethod {
  public:
    DecayMomentumOptimization(MLPerceptron &perceptron, const float lr_0, const float dr,
                              const float mom)
        : initial_lr(lr_0), learning_r(lr_0), momentum(mom), decay_r(dr) {
      setPerceptron(perceptron);
    }

    void setPerceptron(MLPerceptron &perceptron);
    void compute(BackpropStorage &storage) override;
    void update() override;

  private:
    const float initial_lr;
    const float decay_r;
    float learning_r;
    const float momentum;

    size_t epoch = 0;

    std::vector<math::FloatMatrix> old_weight_change;
  };


  class RPropPOptimization : public OptimizationMethod {
  public:
    explicit RPropPOptimization(MLPerceptron &perceptron, const float eta_p = 1.2,
                                const float eta_m = 0.5, const float lr_max = 50.0,
                                const float lr_min = 1e-6)
        : eta_plus(eta_p), eta_minus(eta_m), update_max(lr_max), update_min(lr_min) {
      setPerceptron(perceptron);
    }

    void setPerceptron(MLPerceptron &perceptron);
    void compute(BackpropStorage &storage) override;

  private:
    std::vector<math::FloatMatrix> weights_updates;
    std::vector<math::FloatMatrix> old_gradients;
    std::vector<math::FloatMatrix> weights_changes;
    const float eta_plus;
    const float eta_minus;
    const float update_max;
    const float update_min;
  };
}   // namespace nnet
