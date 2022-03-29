#include "DecayOptimization.hpp"

namespace nnet {

  void DecayOptimization::optimize(BackpropStorage &storage, cl::CommandQueue &queue) {
    auto buf = storage.getGradient().scale(learning_r, queue);
    storage.getWeights().ipsub(1.0f, buf, queue);
  }

  void DecayOptimization::update() {
    epoch++;
    learning_r = (1 / (1 + decay_r * static_cast<float>(epoch))) * initial_lr;
  }

}   // namespace nnet