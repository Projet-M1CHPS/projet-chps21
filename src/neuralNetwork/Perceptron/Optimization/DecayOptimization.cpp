#include "DecayOptimization.hpp"

namespace nnet {

  void DecayOptimization::optimize(BackpropStorage &storage) {
    storage.getWeights() -= (storage.getGradient() * learning_r);
  }

  void DecayOptimization::update() {
    epoch++;
    learning_r = (1 / (1 + decay_r * static_cast<float>(epoch))) * initial_lr;
  }

}   // namespace nnet