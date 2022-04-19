#include "neuralNetwork/CNN/CNNOptimizer.hpp"


namespace nnet {

  CNNOptimizer::CNNOptimizer(CNNModel &model, std::unique_ptr<Optimization> mlpTm)
      : cnn(&model.getCnn()), mlpOptimizer(model.getMlp(), std::move(mlpTm)) {}

  namespace {
    clFTensor forward(const CNN &cnn, const clFTensor &inputs,
                      std::vector<std::unique_ptr<CNNStorageBP>> &storages,
                      cl::CommandQueue &queue) {
      auto &layers = cnn.getLayers();

      clFTensor output = inputs.shallowCopy();

      for (size_t i = 0; i < layers.size(); i++) {
        output = layers[i]->computeForward(output, *storages[i]);
      }

      reorganizeForward(queue, output, inputs.getDepth(), cnn.getTopology().getNBranchFinal());

      return output;
    }

    void backward(const CNN &cnn, const clFTensor &inputs, clFTensor &errorsFlatten,
                  std::vector<std::unique_ptr<CNNStorageBP>> &storages, cl::CommandQueue &queue) {
      auto &layers = cnn.getLayers();

      reorganizeBackward(queue, errorsFlatten, errorsFlatten.getDepth(), cnn.getTopology().getDepth(), layers.back()->getOutputSize());

      clFTensor output = inputs.shallowCopy();

      for (long i = static_cast<long>(layers.size() - 1); i > -1; i--) {
        output = layers[i]->computeForward(output, *storages[i]);
      }
    }
  }   // namespace

  void CNNOptimizer::optimize(const math::clFTensor &inputs, const clFTensor &targets,
                              MLPOptimizer::WeightUpdater &mlpUpdater, cl::CommandQueue &queue) {
    std::vector<std::unique_ptr<CNNStorageBP>> storages = cnn->getTopology().convertToStorage();

    clFTensor flatten = forward(*cnn, inputs, storages, queue);

    clFTensor errorFlatten = mlpOptimizer.optimize(flatten, targets, mlpUpdater, queue);

    backward(*cnn, inputs, errorFlatten, storages, queue);
  }

}   // namespace nnet