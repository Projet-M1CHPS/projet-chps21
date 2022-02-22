#include "SGDOptimization.hpp"

namespace nnet {

  SGDOptimization::SGDOptimization(const MLPerceptron &perceptron, float lr) : learning_r(lr) {}

  void SGDOptimization::optimize(BackpropStorage &storage, utils::clWrapper &wrapper,
                                 cl::CommandQueue &queue) {
    auto buf = storage.getGradient().scale(learning_r, wrapper, queue);
    storage.getWeights().ipsub(buf, wrapper, queue);
  }
}   // namespace nnet