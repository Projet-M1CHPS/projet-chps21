#include "Optimization/RProPOptimization.hpp"

namespace nnet {

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

  void RPropPOptimization::optimize(BackpropStorage &storage, utils::clWrapper& wrapper, cl::CommandQueue& queue) {
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
