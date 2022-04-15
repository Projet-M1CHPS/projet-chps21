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


      // TODO : reshape output with strange function
      queue.finish();

      return output.flatten();
    }

    void backward(const CNN &cnn, const clFTensor &inputs, const clFTensor &errorsFlatten,
                  std::vector<std::unique_ptr<CNNStorageBP>> &storages, cl::CommandQueue &queue) {
      auto &layers = cnn.getLayers();

      clFTensor output = inputs.shallowCopy();

      for (long i = static_cast<long>(layers.size() - 1); i > -1; i--) {
        output = layers[i]->computeForward(output, *storages[i]);
      }

      // TODO : reshape output with strange function
      queue.finish();
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