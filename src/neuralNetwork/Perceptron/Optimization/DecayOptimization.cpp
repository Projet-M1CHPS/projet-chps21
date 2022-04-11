#include "DecayOptimization.hpp"

namespace nnet {

  void DecayOptimization::optimize(math::clFMatrix &gradient, math::clFMatrix &dest, size_t layer,
                                   cl::CommandQueue &queue) {
    dest.ipsub(learning_r, gradient, queue);
  }

  void DecayOptimization::update() {
    epoch++;
    learning_r = (1 / (1 + decay_r * static_cast<float>(epoch))) * initial_lr;
  }

}   // namespace nnet