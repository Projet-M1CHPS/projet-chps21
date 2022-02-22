#include "DecayOptimization.hpp"

namespace nnet {

  void DecayOptimization::optimize(BackpropStorage &storage, utils::clWrapper& wrapper, cl::CommandQueue& queue) {
    auto buf = storage.getGradient().scale(learning_r, wrapper, queue);
    storage.getWeights().ipsub(buf, wrapper, queue);
  }

  void DecayOptimization::update() {
    epoch++;
    learning_r = (1 / (1 + decay_r * static_cast<float>(epoch))) * initial_lr;
  }

}   // namespace nnet