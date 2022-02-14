#include "SGDOptimization.hpp"

namespace nnet {

  SGDOptimization::SGDOptimization(const MLPerceptron &perceptron, float lr) : learning_r(lr) {}

  void SGDOptimization::optimize(BackpropStorage &storage) {
    storage.getWeights() -= (storage.getGradient() * learning_r);
  }
}   // namespace nnet