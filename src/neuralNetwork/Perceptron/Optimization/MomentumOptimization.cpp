#include "MomentumOptimization.hpp"

namespace nnet {

  MomentumOptimization::MomentumOptimization(const MLPerceptron &perceptron,
                                             const float learning_rate, const float momentum)
      : lr(learning_rate), momentum(momentum) {
    auto &topology = perceptron.getTopology();
    for (size_t i = 0; i < topology.size() - 1; i++) {
      old_weight_change.emplace_back(topology[i + 1], topology[i]);
      old_weight_change.back().fill(0.0);
    }
  }

  void MomentumOptimization::optimize(BackpropStorage &storage, utils::clWrapper& wrapper, cl::CommandQueue& queue) {
    auto weight_change =
            (storage.getGradient() * lr) + (old_weight_change[storage.getIndex()] * momentum);
    storage.getWeights() -= weight_change;
    old_weight_change[storage.getIndex()] = std::move(weight_change);
  }

}   // namespace nnet
