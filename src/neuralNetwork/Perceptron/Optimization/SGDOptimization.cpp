#include "SGDOptimization.hpp"

namespace nnet {

  SGDOptimization::SGDOptimization(const MLPerceptron &perceptron, float lr) : learning_r(lr) {}

  void SGDOptimization::optimize(math::clFMatrix &gradient, math::clFMatrix &dest, size_t layer,
                                 cl::CommandQueue &queue) {
    dest.ipsub(learning_r, gradient, queue);
  }
}   // namespace nnet