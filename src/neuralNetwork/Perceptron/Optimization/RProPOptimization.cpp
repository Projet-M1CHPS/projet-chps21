#include "Optimization/RProPOptimization.hpp"

namespace nnet {

  RPropPOptimization::RPropPOptimization(const MLPerceptron &perceptron, const float eta_p,
                                         const float eta_m, const float lr_max, const float lr_min)
      : eta_plus(eta_p), eta_minus(eta_m), update_max(lr_max), update_min(lr_min) {
    auto &topology = perceptron.getTopology();
    cl::CommandQueue queue(utils::cl_wrapper.getDefaultDevice());

    for (size_t i = 0; i < topology.size() - 1; i++) {
      math::FloatMatrix buf(topology[i + 1], topology[i]);
      buf.fill(0.1);
      weights_updates.emplace_back(buf, queue, false);

      math::FloatMatrix buf2(topology[i + 1], topology[i]);
      buf2.fill(0.0);
      old_gradients.emplace_back(buf2, queue, false);

      math::FloatMatrix buf3(topology[i + 1], topology[i]);
      buf3.fill(0.0);
      weights_changes.emplace_back(buf3, queue, false);
      queue.finish();
    }
  }

  void RPropPOptimization::optimize(BackpropStorage &storage, cl::CommandQueue &queue) {
    // Aliases to increase readability
    size_t index = storage.getIndex();
    auto &weights = storage.getWeights();
    auto &gradient = storage.getGradient();
    auto &weights_update = weights_updates[index];
    auto &last_weights_change = weights_changes[index];
    auto &old_gradient = old_gradients[index];

    auto kernel = utils::cl_wrapper.getKernels().getKernel("optimization.cl", "rprop_update");
    kernel.setArg(0, weights.getBuffer());
    kernel.setArg(1, gradient.getBuffer());
    kernel.setArg(2, old_gradient.getBuffer());
    kernel.setArg(3, weights_update.getBuffer());
    kernel.setArg(4, last_weights_change.getBuffer());
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(weights.size()), cl::NullRange);
  }
}   // namespace nnet
