#include "Optimization/Optimization.hpp"

namespace nnet {

  SGDOptimization::SGDOptimization(const MLPerceptron &perceptron, float lr) : learning_r(lr) {}

  void SGDOptimization::optimize(BackpropStorage &storage) {
    storage.getWeights() -= (storage.getGradient() * learning_r);
  }

  void DecayOptimization::optimize(BackpropStorage &storage) {
    storage.getWeights() -= (storage.getGradient() * learning_r);
  }

  void DecayOptimization::update() {
    epoch++;
    learning_r = (1 / (1 + decay_r * static_cast<float>(epoch))) * initial_lr;
  }

  MomentumOptimization::MomentumOptimization(const MLPerceptron &perceptron,
                                             const float learning_rate, const float momentum)
      : lr(learning_rate), momentum(momentum) {
    auto &topology = perceptron.getTopology();
    for (size_t i = 0; i < topology.size() - 1; i++) {
      old_weight_change.emplace_back(topology[i + 1], topology[i]);
      old_weight_change.back().fill(0.0);
    }
  }

  void MomentumOptimization::optimize(BackpropStorage &storage) {
    auto weight_change =
            (storage.getGradient() * lr) + (old_weight_change[storage.getIndex()] * momentum);
    storage.getWeights() -= weight_change;
    old_weight_change[storage.getIndex()] = std::move(weight_change);
  }

  DecayMomentumOptimization::DecayMomentumOptimization(const MLPerceptron &perceptron,
                                                       const float lr_0, const float dr,
                                                       const float mom)
      : initial_lr(lr_0), learning_r(lr_0), momentum(mom), decay_r(dr) {
    auto &topology = perceptron.getTopology();
    for (size_t i = 0; i < topology.size() - 1; i++) {
      old_weight_change.emplace_back(topology[i + 1], topology[i]);
      old_weight_change.back().fill(0.0);
    }
  }

  void DecayMomentumOptimization::optimize(BackpropStorage &storage) {
    auto dw = (storage.getGradient() * learning_r) +
              (old_weight_change[storage.getIndex()] * momentum);
    storage.getWeights() -= dw;
    old_weight_change[storage.getIndex()] = std::move(dw);
  }


  void DecayMomentumOptimization::update() {
    epoch++;
    learning_r = (1 / (1 + decay_r * static_cast<float>(epoch))) * static_cast<float>(initial_lr);
  }

  RPropPOptimization::RPropPOptimization(const MLPerceptron &perceptron, const float eta_p,
                                         const float eta_m, const float lr_max, const float lr_min)
      : eta_plus(eta_p), eta_minus(eta_m), update_max(lr_max), update_min(lr_min) {
    auto &topology = perceptron.getTopology();
    for (size_t i = 0; i < topology.size() - 1; i++) {
      weights_updates.emplace_back(topology[i + 1], topology[i]);
      weights_updates.back().fill(0.1);

      old_gradients.emplace_back(topology[i + 1], topology[i]);
      old_gradients.back().fill(0.0);

      weights_changes.emplace_back(topology[i + 1], topology[i]);
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

  void RPropPOptimization::optimize(BackpropStorage &storage) {
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
}   // namespace nnet
