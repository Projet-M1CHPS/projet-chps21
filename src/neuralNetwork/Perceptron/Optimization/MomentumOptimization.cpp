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

  void MomentumOptimization::optimize(math::clFMatrix &gradient, math::clFMatrix &dest,
                                      size_t layer, cl::CommandQueue &queue) {
    auto buf = gradient.scale(lr, queue);
    buf.ipadd(momentum, old_weight_change[layer], queue);

    dest.ipsub(1.0f, buf, queue);
    old_weight_change[layer] = std::move(buf);
  }

}   // namespace nnet
