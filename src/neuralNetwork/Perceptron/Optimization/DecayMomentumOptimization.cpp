#include "DecayMomentumOptimization.hpp"

namespace nnet {

  DecayMomentumOptimization::DecayMomentumOptimization(const MLPerceptron &perceptron,
                                                       utils::clWrapper &wrapper, const float lr_0,
                                                       const float dr, const float mom)
      : initial_lr(lr_0), learning_r(lr_0), momentum(mom), decay_r(dr) {
    auto &topology = perceptron.getTopology();
    for (size_t i = 0; i < topology.size() - 1; i++) {
      math::FloatMatrix buf(topology[i + 1], topology[i]);
      buf.fill(0.0f);
      old_weight_change.emplace_back(buf, wrapper);
    }
  }

  void DecayMomentumOptimization::optimize(BackpropStorage &storage, utils::clWrapper &wrapper,
                                           cl::CommandQueue &queue) {
    auto buf = storage.getGradient().scale(learning_r, wrapper, queue);
    buf.ipadd(old_weight_change[storage.getIndex()], wrapper, queue);

    storage.getWeights().ipsub(buf, wrapper, queue);
    old_weight_change[storage.getIndex()] = std::move(buf);
  }


  void DecayMomentumOptimization::update() {
    epoch++;
    learning_r = (1 / (1 + decay_r * static_cast<float>(epoch))) * static_cast<float>(initial_lr);
  }

}   // namespace nnet