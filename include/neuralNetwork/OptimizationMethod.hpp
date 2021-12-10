#pragma once

#include <vector>

#include "BackpropStorage.hpp"
#include "Matrix.hpp"


namespace nnet {

  /** Keep this in sync with new classes
   *
   */
  enum class OptimizationAlgorithm { standard, decay, momentum, rpropPlus };

  template<typename T>
  class OptimizationMethod {
  public:
    OptimizationMethod() = default;
    virtual ~OptimizationMethod() = default;

    virtual void compute(BackpropStorage<T> &storage) = 0;
    virtual void update(){};
  };

  template<typename T>
  class SGDOptimization : public OptimizationMethod<T> {
  public:
    SGDOptimization(const T learningRate) : learningRate(learningRate) {}

    void compute(BackpropStorage<T> &storage) override {
      storage.getWeights() -= (storage.getGradient() * learningRate);
    }

  private:
    const T learningRate;
  };


  template<typename T>
  class DecayOptimization : public OptimizationMethod<T> {
  public:
    DecayOptimization(const T lr_0, const T dr)
        : initial_lr(lr_0), decay_r(dr), curr_lr(lr_0), epoch(0) {}

    void compute(BackpropStorage<T> &storage) override {
      storage.getWeights() -= (storage.getGradient() * curr_lr);
    }

    void update() override {
      epoch++;
      curr_lr = (1 / (1 + decay_r * epoch)) * static_cast<T>(initial_lr);
    }

  private:
    const T initial_lr;
    const T decay_r;
    T curr_lr;

    size_t epoch;
  };


  template<typename T>
  class MomentumOptimization : public OptimizationMethod<T> {
  public:
    MomentumOptimization(const std::vector<size_t> &topology, const T learning_rate,
                         const T momentum)
        : lr(learning_rate), momentum(momentum) {
      for (size_t i = 0; i < topology.size() - 1; i++) {
        old_weight_change.push_back(math::Matrix<T>(topology[i + 1], topology[i]));
        old_weight_change.back().fill(0.0);
      }
    }

    void compute(BackpropStorage<T> &storage) {
      auto weight_change =
              (storage.getGradient() * lr) + (old_weight_change[storage.getIndex()] * momentum);
      storage.getWeights() -= weight_change;
      old_weight_change[storage.getIndex()] = std::move(weight_change);
    }

  private:
    const T lr;
    const T momentum;
    std::vector<math::Matrix<T>> old_weight_change;
  };


  template<typename T>
  class DecayMomentumOptimization : public OptimizationMethod<T> {
  public:
    DecayMomentumOptimization(const std::vector<size_t> &topology, const T lr_0, const T dr,
                              const T mom)
        : learningRate_0(lr_0), learningRate(lr_0), momentum(mom), decayRate(dr) {
      for (size_t i = 0; i < topology.size() - 1; i++) {
        dw_old.push_back(math::Matrix<T>(topology[i + 1], topology[i]));
        dw_old.back().fill(0.0);
      }
    }

    void compute(BackpropStorage<T> &storage) {
      auto dw = (storage.getGradient() * learningRate) + (dw_old[storage.getIndex()] * momentum);
      storage.getWeights() -= dw;
      dw_old[storage.getIndex()] = std::move(dw);
    }

    void update() {
      epoch++;
      learningRate = (1 / (1 + decayRate * epoch)) * static_cast<T>(learningRate_0);
    }

  private:
    const T learningRate_0;
    const T decayRate;
    T learningRate;
    const T momentum;

    size_t epoch = 0;

    std::vector<math::Matrix<T>> dw_old;
  };


  template<typename T>
  class RPropPOptimization : public OptimizationMethod<T> {
  public:
    RPropPOptimization(const std::vector<size_t> &topology, const T eta_p = 1.2,
                       const T eta_m = 0.5, const T lr_max = 50.0, const T lr_min = 1e-6)
        : eta_plus(eta_p), eta_minus(eta_m), update_max(lr_max), update_min(lr_min) {
      for (size_t i = 0; i < topology.size() - 1; i++) {
        weights_updates.push_back(math::Matrix<T>(topology[i + 1], topology[i]));
        weights_updates.back().fill(0.1);

        old_gradients.push_back(math::Matrix<T>(topology[i + 1], topology[i]));
        old_gradients.back().fill(0.0);

        weights_changes.push_back(math::Matrix<T>(topology[i + 1], topology[i]));
        weights_changes.back().fill(0.0);
      }
    }

    T sign(const T x) {
      if (std::abs(x) < 1e-6) {
        return 0;
      } else if (x > 0) {
        return 1;
      }
      return -1;
    }

    void compute(BackpropStorage<T> &storage) {
      // Aliases to increase readability
      size_t index = storage.getIndex();
      auto &weights = storage.getWeights();
      auto &gradient = storage.getGradient();
      auto &weights_update = weights_updates[index];
      auto &last_weights_change = weights_changes[index];
      auto &old_gradient = old_gradients[index];

      for (size_t i = 0; i < weights.getRows(); i++) {
        for (size_t j = 0; j < weights.getCols(); j++) {
          T weight_change = 0.0;
          T change = sign(gradient(i, j) * old_gradients[index](i, j));

          // The gradient is converging in the same direction as the previous update
          // Increase the delta to converge faster
          if (change > 0.0) {
            T delta = std::min(weights_update(i, j) * eta_plus, update_max);
            weight_change = delta * sign(gradient(i, j));
            weights_update(i, j) = delta;
            old_gradient(i, j) = gradient(i, j);
          }
          // The gradient as changed direction
          // Rollback and reduce the delta to converge slower
          else if (change < 0) {
            T delta = std::max(weights_update(i, j) * eta_minus, update_min);
            weights_update(i, j) = delta;
            weight_change = -last_weights_change(i, j);
            old_gradient(i, j) = 0.0;
          }
          // No need to change the delta
          else if (change == 0) {
            T delta = weights_update(i, j);
            weight_change = sign(gradient(i, j)) * delta;
            old_gradient(i, j) = gradient(i, j);
          }
          weights(i, j) -= weight_change;
          last_weights_change(i, j) = weight_change;
        }
      }
    }

  private:
    std::vector<math::Matrix<T>> weights_updates;
    std::vector<math::Matrix<T>> old_gradients;
    std::vector<math::Matrix<T>> weights_changes;
    const T eta_plus;
    const T eta_minus;
    const T update_max;
    const T update_min;
  };

}   // namespace nnet
