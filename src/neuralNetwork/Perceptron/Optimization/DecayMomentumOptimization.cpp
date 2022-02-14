#include "DecayMomentumOptimization.hpp"

namespace nnet {

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

}   // namespace nnet