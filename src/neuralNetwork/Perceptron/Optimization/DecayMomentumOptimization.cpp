#include "DecayMomentumOptimization.hpp"

namespace nnet {

  DecayMomentumOptimization::DecayMomentumOptimization(const MLPerceptron &perceptron,
                                                       const float lr_0, const float dr,
                                                       const float mom)
      : initial_lr(lr_0), decay_r(dr), learning_r(lr_0), momentum(mom) {
    auto &topology = perceptron.getTopology();
    for (size_t i = 0; i < topology.size() - 1; i++) {
      math::FloatMatrix buf(topology[i + 1], topology[i]);
      buf.fill(0.0f);
      old_weight_change.emplace_back(buf);
    }
  }

  void DecayMomentumOptimization::optimize(math::clFMatrix &gradient, math::clFMatrix &dest,
                                           size_t layer, cl::CommandQueue &queue) {
    auto buf = gradient.scale(learning_r, queue);
    buf.ipadd(momentum, old_weight_change[layer], queue);

    dest.ipsub(1.0f, buf, queue);
    old_weight_change[layer] = std::move(buf);
  }


  void DecayMomentumOptimization::update() {
    epoch++;
    learning_r = (1 / (1 + decay_r * static_cast<float>(epoch))) * static_cast<float>(initial_lr);
  }

}   // namespace nnet