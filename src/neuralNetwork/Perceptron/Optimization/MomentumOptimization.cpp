#include "MomentumOptimization.hpp"

namespace nnet {

  MomentumOptimization::MomentumOptimization(const MLPerceptron &perceptron,
                                             const float learning_rate, const float momentum)
      : lr(learning_rate), momentum(momentum) {
    auto &topology = perceptron.getTopology();
    for (size_t i = 0; i < topology.size() - 1; i++) {
      math::FloatMatrix buf(topology[i + 1], topology[i]);
      buf.fill(0.0f);

      old_weight_change.emplace_back(buf);
    }
  }

  void MomentumOptimization::optimize(BackpropStorage &storage, cl::CommandQueue &queue) {
    auto buf = storage.getGradient().scale(lr, queue);
    buf.ipadd(1.0f, old_weight_change[storage.getIndex()], queue);

    storage.getWeights().ipsub(1.0f, buf, queue);
    old_weight_change[storage.getIndex()] = std::move(buf);
  }

}   // namespace nnet
