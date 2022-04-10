#include "neuralNetwork/CNN/CNNOptimizer.hpp"


namespace nnet {

  CNNOptimizer::CNNOptimizer(CNNModel &model) : cnn(&model.getCnn()), mlp(&model.getMlp()) {}


  void CNNOptimizer::optimize(const math::clFTensor &inputs, const clFTensor &targets) {

    std::vector<std::unique_ptr<CNNStorageBP>> storages = cnn->getTopology().convertToStorage();

    clFTensor flatten = forward(inputs, storages);

    // TODO : calculter l'erreur sur le flatten
    clFTensor errorFlatten;

    backward(inputs, errorFlatten, storages);
  }

  clFTensor CNNOptimizer::forward(const clFTensor &inputs, std::vector<std::unique_ptr<CNNStorageBP>>& storages) {
    auto &layers = cnn->getLayers();
    if (layers.empty()) { throw std::runtime_error("no layer in cnn for optimization"); }

    clFTensor output = inputs.shallowCopy();

    for (auto &layer : layers) {
      output = layer->computeForward(output, *storages[0]);
    }
    utils::cl_wrapper.getDefaultQueue().finish();

    return output.flatten();
  }

  void CNNOptimizer::backward(const clFTensor &inputs, const clFTensor &errorsFlatten, std::vector<std::unique_ptr<CNNStorageBP>>& storages) {

  }

}   // namespace nnet