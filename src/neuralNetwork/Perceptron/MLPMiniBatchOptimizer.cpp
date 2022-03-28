#include "MLPMiniBatchOptimizer.hpp"

namespace nnet {

  MLPMiniBatchOptimizer::MLPMiniBatchOptimizer(MLPModel &model, std::unique_ptr<Optimization> tm,
                                               size_t batch_size)
      : MLPBatchOptimizer(model, std::move(tm)), batch_size(batch_size) {}


  void MLPMiniBatchOptimizer::optimize(const std::vector<math::clFMatrix> &inputs,
                                       const std::vector<math::clFMatrix> &targets) {
    size_t n = inputs.size();

    cl::CommandQueue queue(utils::cl_wrapper.getDefaultDevice());
    cl::Kernel zero_out = utils::cl_wrapper.getKernels().getKernel("utils.cl", "zero_out");

    auto zerol = [&](auto &mat) {
      zero_out.setArg(0, mat);
      queue.enqueueNDRangeKernel(zero_out, cl::NullRange,
                                 cl::NDRange(mat.getRows(), mat.getCols()));
    };

    for (size_t i = 0; i < n; i += batch_size) {
      std::for_each(avg_gradients.begin(), avg_gradients.end(), zerol);
      std::for_each(avg_errors.begin(), avg_errors.end(), zerol);

      // If n is not a multiple of batch_size, the last batch will be smaller
      size_t curr_batch_size = std::min(batch_size, n - i);

      // Compute the average gradient and error for the current batch
      for (size_t j = 0; j < curr_batch_size; j++) {
        forward(inputs[i + j], queue);
        storage.getError() = layers_af[layers_af.size() - 1].sub(1.0f, targets[i + j], queue);
        computeGradient(queue);
      }

      for (auto &it : avg_gradients) {
        it.ipscale(1.0f / static_cast<float>(curr_batch_size), queue);
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
