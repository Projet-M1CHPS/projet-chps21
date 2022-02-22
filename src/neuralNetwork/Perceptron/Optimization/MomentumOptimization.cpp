#include "MomentumOptimization.hpp"

namespace nnet {

  MomentumOptimization::MomentumOptimization(const MLPerceptron &perceptron,
                                             utils::clWrapper &wrapper, const float learning_rate,
                                             const float momentum)
      : lr(learning_rate), momentum(momentum) {
    auto &topology = perceptron.getTopology();
    for (size_t i = 0; i < topology.size() - 1; i++) {
      math::FloatMatrix buf(topology[i + 1], topology[i]);
      buf.fill(0.0f);

      old_weight_change.emplace_back(buf, wrapper);
    }
  }

  void MomentumOptimization::optimize(BackpropStorage &storage, utils::clWrapper &wrapper,
                                      cl::CommandQueue &queue) {
    auto buf = storage.getGradient().scale(lr, wrapper, queue);
    buf.ipadd(old_weight_change[storage.getIndex()], wrapper, queue);

    storage.getWeights().ipsub(buf, wrapper, queue);
    old_weight_change[storage.getIndex()] = std::move(buf);
  }

}   // namespace nnet
