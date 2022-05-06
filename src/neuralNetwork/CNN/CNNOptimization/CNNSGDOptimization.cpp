#include "CNNSGDOptimization.hpp"

namespace nnet {

  void CNNSGDOptimization::optimize(const math::clFTensor &gradient, math::clFTensor &dest,
                                    cl::CommandQueue &queue) {
    // TODO : Check si on le fait en bloquant
    dest.ipadd(-learning_r, gradient, queue, true);
  }

}   // namespace nnet
