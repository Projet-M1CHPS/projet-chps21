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

    void compute(BackpropStorage &storage) override {
      storage.getWeights() -= (storage.getGradient() * learning_r);
    }

  private:
    const float learning_r;
  };

  class DecayOptimization : public OptimizationMethod {
  public:
    DecayOptimization(const float lr_0, const float dr)
        : initial_lr(lr_0), decay_r(dr), curr_lr(lr_0), epoch(0) {}

    void compute(BackpropStorage &storage) override {
      storage.getWeights() -= (storage.getGradient() * curr_lr);
    }

    void update() override {
      epoch++;
      curr_lr = (1 / (1 + decay_r * epoch)) * initial_lr;
    }

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

    void setPerceptron(MLPerceptron &perceptron) {
      old_weight_change.clear();

      auto &topology = perceptron.getTopology();
      for (size_t i = 0; i < topology.size() - 1; i++) {
        old_weight_change.push_back(math::FloatMatrix (topology[i + 1], topology[i]));
        old_weight_change.back().fill(0.0);
      }
    }

    void compute(BackpropStorage &storage) {
      auto weight_change =
              (storage.getGradient() * lr) + (old_weight_change[storage.getIndex()] * momentum);
      storage.getWeights() -= weight_change;
      old_weight_change[storage.getIndex()] = std::move(weight_change);
    }


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

    void setPerceptron(MLPerceptron &perceptron) {
      old_weight_change.clear();

      auto &topology = perceptron.getTopology();
      for (size_t i = 0; i < topology.size() - 1; i++) {
        old_weight_change.push_back(math::FloatMatrix(topology[i + 1], topology[i]));
        old_weight_change.back().fill(0.0);
      }
    }


    void compute(BackpropStorage &storage) override {
      auto dw = (storage.getGradient() * learning_r) +
                (old_weight_change[storage.getIndex()] * momentum);
      storage.getWeights() -= dw;
      old_weight_change[storage.getIndex()] = std::move(dw);
    }

    void update() override {
      epoch++;
      learning_r = (1 / (1 + decay_r * epoch)) * static_cast<float>(initial_lr);
    }

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

    void setPerceptron(MLPerceptron &perceptron) {
      weights_updates.clear();
      old_gradients.clear();
      weights_changes.clear();

      auto &topology = perceptron.getTopology();
      for (size_t i = 0; i < topology.size() - 1; i++) {
        weights_updates.push_back(math::FloatMatrix (topology[i + 1], topology[i]));
        weights_updates.back().fill(0.1);

        old_gradients.push_back(math::FloatMatrix(topology[i + 1], topology[i]));
        old_gradients.back().fill(0.0);

        weights_changes.push_back(math::FloatMatrix(topology[i + 1], topology[i]));
        weights_changes.back().fill(0.0);
      }
    }

    float sign(const float x) {
      if (std::abs(x) < 1e-6) {
        return 0;
      } else if (x > 0) {
        return 1;
      }
      return -1;
    }

    void compute(BackpropStorage &storage) override {
      // Aliases to increase readability
      size_t index = storage.getIndex();
      auto &weights = storage.getWeights();
      auto &gradient = storage.getGradient();
      auto &weights_update = weights_updates[index];
      auto &last_weights_change = weights_changes[index];
      auto &old_gradient = old_gradients[index];

      for (size_t i = 0; i < weights.getRows(); i++) {
        for (size_t j = 0; j < weights.getCols(); j++) {
          float weight_change = 0.0;
          float change = sign(gradient(i, j) * old_gradient(i, j));

          // The gradient is converging in the same direction as the previous update
          // Increase the delta to converge faster
          if (change > 0.0) {
            float delta = std::min(weights_update(i, j) * eta_plus, update_max);
            weight_change = delta * sign(gradient(i, j));
            weights_update(i, j) = delta;
            old_gradient(i, j) = gradient(i, j);
          }
          // The gradient as changed direction
          // Rollback and reduce the delta to converge slower
          else if (change < 0) {
            float delta = std::max(weights_update(i, j) * eta_minus, update_min);
            weights_update(i, j) = delta;
            weight_change = -last_weights_change(i, j);
            old_gradient(i, j) = 0.0;
          }
          // No need to change the delta
          else if (change == 0) {
            float delta = weights_update(i, j);
            weight_change = sign(gradient(i, j)) * delta;
            old_gradient(i, j) = gradient(i, j);
          }
          weights(i, j) -= weight_change;
          last_weights_change(i, j) = weight_change;
        }
      }
    }

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
