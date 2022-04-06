#include "SGDOptimization.hpp"

namespace nnet {

  SGDOptimization::SGDOptimization(const MLPerceptron &perceptron, float lr) : learning_r(lr) {}

  void SGDOptimization::optimize(BackpropStorage &storage, cl::CommandQueue &queue) {
    storage.getWeights().ipsub(learning_r, storage.getGradient(), queue);
  }
}   // namespace nnet