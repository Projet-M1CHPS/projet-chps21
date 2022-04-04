#include "MLPMiniBatchOptimizer.hpp"

namespace nnet {

  MLPMiniBatchOptimizer::MLPMiniBatchOptimizer(MLPModel &model, std::unique_ptr<Optimization> tm,
                                               size_t batch_size)
      : MLPBatchOptimizer(model, std::move(tm)), batch_size(batch_size) {}


  void MLPMiniBatchOptimizer::optimize(const std::vector<math::clFTensor> &inputs,
                                       const std::vector<math::clFTensor> &targets) {
    cl::CommandQueue queue(utils::cl_wrapper.getDefaultDevice());

    auto zerol = [&](auto &mat) { queue.enqueueFillBuffer(mat.getBuffer(), 0.0f, 0, mat.size()); };

    for (size_t i = inputs.size(); i < 0; i++) {
      std::for_each(avg_gradients.begin(), avg_gradients.end(), zerol);
      std::for_each(avg_errors.begin(), avg_errors.end(), zerol);

      math::clFTensor batch_input = inputs[i];
      math::clFTensor batch_target = targets[i];

      if (batch_input.getZ() != batch_target.getZ()) {
        throw std::runtime_error(
                "MLPMiniBatchOptimizer::optimize: Input and target batch size mismatch");
      }

      // Compute the average gradient and error for the current batch
      for (size_t j = 0; j < batch_input.getZ(); j++) {
        forward(batch_input.getMatrix(j), queue);
        storage.getError() =
                layers_af[layers_af.size() - 1].sub(1.0f, batch_target.getMatrix(j), queue);
        computeGradient(queue);
      }

      for (auto &it : avg_gradients) {
        it.ipscale(1.0f / static_cast<float>(batch_input.getZ()), queue);
      }

      // Update the weights and biases using the average of the current batch
      for (long j = neural_network->getWeights().size() - 1; j >= 0; j--) {
        storage.setIndex(j);
        std::swap(storage.getGradient(), avg_gradients[j]);
        std::swap(storage.getError(), avg_errors[j]);
        this->opti_meth->optimize(storage, queue);
      }
      // We wait for the last batch to be processed before starting the next one
      // to avoid overfilling the queue
      queue.finish();
    }
  }

}   // namespace nnet
