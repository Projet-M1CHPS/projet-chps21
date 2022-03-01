#include "SGDOptimization.hpp"

namespace nnet {

  SGDOptimization::SGDOptimization(const MLPerceptron &perceptron, utils::clWrapper &wrapper,
                                   float lr)
      : learning_r(lr) {}

  void SGDOptimization::optimize(BackpropStorage &storage, utils::clWrapper &wrapper,
                                 cl::CommandQueue &queue) {
    storage.getWeights().ipsub(learning_r, storage.getGradient(), wrapper, queue);
  }
}   // namespace nnet