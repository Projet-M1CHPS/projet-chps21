#include "neuralNetwork/CNN/CNNOptimizer.hpp"


namespace nnet {

  CNNOptimizer::CNNOptimizer(CNNModel &model, std::unique_ptr<Optimization> mlpTm)
      : cnn(&model.getCnn()), mlpOptimizer(model.getMlp(), std::move(mlpTm)) {}

  namespace {
    clFTensor forward(const CNN &cnn, const clFTensor &inputs,
                      std::vector<std::unique_ptr<CNNStorageBP>> &storages,
                      cl::CommandQueue &queue) {
      auto &layers = cnn.getLayers();
      if (layers.empty()) { throw std::runtime_error("no layer in cnn for optimization"); }

      clFTensor output = inputs.shallowCopy();

      for (auto &layer : layers) { output = layer->computeForward(output, *storages[0]); }
      utils::cl_wrapper.getDefaultQueue().finish();

      return output.flatten();
    }

    void backward(const CNN &cnn, const clFTensor &inputs, const clFTensor &errorsFlatten,
                  std::vector<std::unique_ptr<CNNStorageBP>> &storages, cl::CommandQueue &queue) {}
  }   // namespace

  void CNNOptimizer::optimize(const math::clFTensor &inputs, const clFTensor &targets,
                              MLPOptimizer::WeightUpdater &mlpUpdater, cl::CommandQueue &queue) {
    std::vector<std::unique_ptr<CNNStorageBP>> storages = cnn->getTopology().convertToStorage();

    clFTensor flatten = forward(*cnn, inputs, storages, queue);

    clFTensor errorFlatten = mlpOptimizer.optimize(flatten, targets, mlpUpdater, queue);

    backward(*cnn, inputs, errorFlatten, storages, queue);
  }

}   // namespace nnet