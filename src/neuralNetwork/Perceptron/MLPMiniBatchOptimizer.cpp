#include "MLPMiniBatchOptimizer.hpp"

namespace nnet {

  MLPMiniBatchOptimizer::MLPMiniBatchOptimizer(MLPModel &model, std::shared_ptr<Optimization> tm,
                                               size_t batch_size)
      : MLPBatchOptimizer(model, std::move(tm)), batch_size(batch_size) {}


  void MLPMiniBatchOptimizer::optimize(const std::vector<math::FloatMatrix> &inputs,
                                       const std::vector<math::FloatMatrix> &targets) {
    size_t n = inputs.size();

    for (size_t i = 0; i < n; i += batch_size) {
      auto mat_reset = [](math::FloatMatrix &m) { m.fill(0); };
      std::for_each(avg_gradients.begin(), avg_gradients.end(), mat_reset);
      std::for_each(avg_errors.begin(), avg_errors.end(), mat_reset);

      // If n is not a multiple of batch_size, the last batch will be smaller
      size_t curr_batch_size = std::min(batch_size, n - i);

      // Compute the average gradient and error for the current batch
      for (size_t j = 0; j < curr_batch_size; j++) {
        forward(inputs[i + j]);
        storage.getError() = layers_af[layers_af.size() - 1] - targets[i + j];
        computeGradient();
      }

      for (auto &it : avg_gradients) { it *= 1.0f / static_cast<float>(curr_batch_size); }

      // Update the weights and biases using the average of the current batch
      for (long j = neural_network->getWeights().size() - 1; j >= 0; j--) {
        storage.setIndex(j);
        storage.getGradient() = avg_gradients[j];
        storage.getError() = avg_errors[j];
        this->opti_meth->optimize(storage);
      }
    }
  }

}   // namespace nnet
