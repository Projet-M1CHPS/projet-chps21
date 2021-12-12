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

  template<typename real = float>
  class OptimizationMethod {
  public:
    OptimizationMethod() = default;
    virtual ~OptimizationMethod() = default;

    virtual void compute(BackpropStorage<real> &storage) = 0;
    virtual void update(){};
  };

  template<typename real = float>
  class SGDOptimization : public OptimizationMethod<real> {
  public:
    SGDOptimization(const real learningRate) : learning_r(learningRate) {}

    void compute(BackpropStorage<real> &storage) override {
      storage.getWeights() -= (storage.getGradient() * learning_r);
    }

  private:
    const real learning_r;
  };


  template<typename real = float>
  class DecayOptimization : public OptimizationMethod<real> {
  public:
    DecayOptimization(const real lr_0, const real dr)
        : initial_lr(lr_0), decay_r(dr), curr_lr(lr_0), epoch(0) {}

    void compute(BackpropStorage<real> &storage) override {
      storage.getWeights() -= (storage.getGradient() * curr_lr);
    }

    void update() override {
      epoch++;
      curr_lr = (1 / (1 + decay_r * epoch)) * initial_lr;
    }

  private:
    const real initial_lr;
    const real decay_r;
    real curr_lr;

    size_t epoch = 0;
  };


  template<typename real = float>
  class MomentumOptimization : public OptimizationMethod<real> {
  public:
    MomentumOptimization(const MLPTopology &topology, const real learning_rate, const real momentum)
        : lr(learning_rate), momentum(momentum) {
      for (size_t i = 0; i < topology.size() - 1; i++) {
        old_weight_change.push_back(math::Matrix<real>(topology[i + 1], topology[i]));
        old_weight_change.back().fill(0.0);
      }
    }

    void compute(BackpropStorage<real> &storage) {
      auto weight_change =
              (storage.getGradient() * lr) + (old_weight_change[storage.getIndex()] * momentum);
      storage.getWeights() -= weight_change;
      old_weight_change[storage.getIndex()] = std::move(weight_change);
    }

  private:
    const real lr;
    const real momentum;
    std::vector<math::Matrix<real>> old_weight_change;
  };


  template<typename real = float>
  class DecayMomentumOptimization : public OptimizationMethod<real> {
  public:
    DecayMomentumOptimization(const MLPTopology &topology, const real lr_0, const real dr,
                              const real mom)
        : initial_lr(lr_0), learning_r(lr_0), momentum(mom), decay_r(dr) {
      for (size_t i = 0; i < topology.size() - 1; i++) {
        old_weight_change.push_back(math::Matrix<real>(topology[i + 1], topology[i]));
        old_weight_change.back().fill(0.0);
      }
    }

    void compute(BackpropStorage<real> &storage) {
      auto dw = (storage.getGradient() * learning_r) +
                (old_weight_change[storage.getIndex()] * momentum);
      storage.getWeights() -= dw;
      old_weight_change[storage.getIndex()] = std::move(dw);
    }

    void update() {
      epoch++;
      learning_r = (1 / (1 + decay_r * epoch)) * static_cast<real>(initial_lr);
    }

  private:
    const real initial_lr;
    const real decay_r;
    real learning_r;
    const real momentum;

    size_t epoch = 0;

    std::vector<math::Matrix<real>> old_weight_change;
  };


  template<typename real = float>
  class RPropPOptimization : public OptimizationMethod<real> {
  public:
    RPropPOptimization(const MLPTopology &topology, const real eta_p = 1.2, const real eta_m = 0.5,
                       const real lr_max = 50.0, const real lr_min = 1e-6)
        : eta_plus(eta_p), eta_minus(eta_m), update_max(lr_max), update_min(lr_min) {
      for (size_t i = 0; i < topology.size() - 1; i++) {
        weights_updates.push_back(math::Matrix<real>(topology[i + 1], topology[i]));
        weights_updates.back().fill(0.1);

        old_gradients.push_back(math::Matrix<real>(topology[i + 1], topology[i]));
        old_gradients.back().fill(0.0);

        weights_changes.push_back(math::Matrix<real>(topology[i + 1], topology[i]));
        weights_changes.back().fill(0.0);
      }
    }

    real sign(const real x) {
      if (std::abs(x) < 1e-6) {
        return 0;
      } else if (x > 0) {
        return 1;
      }
      return -1;
    }

    void compute(BackpropStorage<real> &storage) {
      // Aliases to increase readability
      size_t index = storage.getIndex();
      auto &weights = storage.getWeights();
      auto &gradient = storage.getGradient();
      auto &weights_update = weights_updates[index];
      auto &last_weights_change = weights_changes[index];
      auto &old_gradient = old_gradients[index];

      for (size_t i = 0; i < weights.getRows(); i++) {
        for (size_t j = 0; j < weights.getCols(); j++) {
          real weight_change = 0.0;
          real change = sign(gradient(i, j) * old_gradient(i, j));

          // The gradient is converging in the same direction as the previous update
          // Increase the delta to converge faster
          if (change > 0.0) {
            real delta = std::min(weights_update(i, j) * eta_plus, update_max);
            weight_change = delta * sign(gradient(i, j));
            weights_update(i, j) = delta;
            old_gradient(i, j) = gradient(i, j);
          }
          // The gradient as changed direction
          // Rollback and reduce the delta to converge slower
          else if (change < 0) {
            real delta = std::max(weights_update(i, j) * eta_minus, update_min);
            weights_update(i, j) = delta;
            weight_change = -last_weights_change(i, j);
            old_gradient(i, j) = 0.0;
          }
          // No need to change the delta
          else if (change == 0) {
            real delta = weights_update(i, j);
            weight_change = sign(gradient(i, j)) * delta;
            old_gradient(i, j) = gradient(i, j);
          }
          weights(i, j) -= weight_change;
          last_weights_change(i, j) = weight_change;
        }
      }
    }

  private:
    std::vector<math::Matrix<real>> weights_updates;
    std::vector<math::Matrix<real>> old_gradients;
    std::vector<math::Matrix<real>> weights_changes;
    const real eta_plus;
    const real eta_minus;
    const real update_max;
    const real update_min;
  };

}   // namespace nnet