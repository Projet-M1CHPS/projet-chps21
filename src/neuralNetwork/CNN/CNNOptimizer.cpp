#include "neuralNetwork/CNN/CNNOptimizer.hpp"


namespace nnet {

  CNNOptimizer::CNNOptimizer(CNNModel &model)
      : nn_cnn(&model.getCnn()), nn_mlp(&model.getMlp()) {
    flatten = model.getFlatten().toFloatMatrix(true);

    const auto &layers = nn_cnn->getLayers();
    const auto &topology = nn_cnn->getTopology();
    const size_t deepth = topology.getDeepth();

    // Allocate storage for each layer
    storage.resize(layers.size());
    storage[0].resize(layers[0].size());
    for (size_t i = 1; i < deepth; i++) {
      storage[i].resize(topology(i)->getFeatures() * layers[i - 1].size());
    }

    // Initialize storage
    std::pair<size_t, size_t> inputSize(topology.getInputSize());
    for (size_t i = 0; i < topology(0)->getFeatures(); i++) {
      storage[0][i] = topology(0)->createStorage(inputSize);
    }

    for (size_t i = 1; i < deepth; i++) {
      inputSize = std::make_pair(storage[i - 1].front()->output.getRows(),
                                 storage[i - 1].front()->output.getCols());
      for (size_t j = 0; j < layers[i].size(); j++) {
        storage[i][j] = topology(i)->createStorage(inputSize);
      }
    }
  }

  void CNNStochOptimizer::optimize(const math::clFMatrix &input, const math::clFMatrix &target) {
    // TODO: Use clFMatrix instead
    auto tmp_input = input.toFloatMatrix(true);
    auto tmp_target = input.toFloatMatrix(true);

    forward(tmp_input);

    clFMatrix tmp_flatten(flatten, true);

    FloatMatrix errorFlatten = mlp_opti.optimize(tmp_flatten, target).toFloatMatrix(true);


    /*for (size_t i = 0; i < storage.size(); i++) {
      for (size_t j = 0; j < storage[i].size(); j++) {
        std::cout << storage[i][j]->output << std::endl;
      }
      std::cout << std::endl;
    }*/

    // for(auto& i : errorFlatten) i = 1.0f;

    std::cout << "error flatten \n" << errorFlatten << std::endl;

    backward(tmp_input, errorFlatten);
  }

  void CNNStochOptimizer::optimize(const std::vector<math::clFMatrix> &inputs,
                                   const std::vector<math::clFMatrix> &targets) {
    if (inputs.size() != targets.size()) {
      throw std::runtime_error("Invalid number of inputs or targets");
    }
    for (size_t i = 0; i < inputs.size(); ++i) { optimize(inputs[i], targets[i]); }
  }

  void CNNStochOptimizer::forward(math::FloatMatrix const &input) {
    auto &layers = nn_cnn->getLayers();
    auto &topology = nn_cnn->getTopology();

    for (size_t i = 0; i < topology(0)->getFeatures(); i++) {
      clFMatrix tmp = storage[0][i]->output;
      layers[0][i]->compute(input, tmp);
      storage[0][i]->output = tmp.toFloatMatrix(true);
    }

    for (size_t i = 1; i < topology.getDeepth(); i++) {
      size_t l = 0;
      for (size_t j = 0; j < layers[i - 1].size(); j++) {
        for (size_t k = 0; k < topology(i)->getFeatures(); k++) {
          layers[i][j]->computeForward(storage[i - 1][j]->output, *storage[i][l++]);
        }
      }
    }

    size_t index = 0;

    for (auto &storageElement : storage.back()) {
      for (auto val : storageElement->output) {
        flatten(index, 0) = val;
        index++;
      }
    }
  }

  void CNNStochOptimizer::backward(math::FloatMatrix const &input,
                                   math::FloatMatrix const &errorFlatten) {
    std::cout << "backward" << std::endl;
    // Convert flatten to matrix
    size_t index = 0;
    for (auto &storageElement : storage.back()) {
      for (auto &val : storageElement->output) {
        val = errorFlatten(index, 0);
        std::cout << errorFlatten(index, 0) << std::endl;
        index++;
      }
    }

    std::cout << "convert " << storage.back().size() << std::endl;
    std::cout << storage.back()[0]->output << std::endl;

    auto &layers = nn_cnn->getLayers();
    auto &topology = nn_cnn->getTopology();

    for (long i = storage.size() - 2; i >= -1; i--) {
      size_t l = 0;
      for (size_t j = 0; j < storage[i].size(); j++) {
        const size_t bufferL = l;
        for (size_t k = 0; k < topology(i + 1)->getFeatures(); k++) {
          std::cout << "\n[" << i << ", " << j << "] [" << i + 1 << ", " << l << "]" << std::endl;
          std::cout << storage[i + 1][l]->output << std::endl;
          layers[i + 1][j]->computeBackward(storage[i][j]->output, *storage[i + 1][l++]);
          std::cout << storage[i][j]->output << std::endl;
        }
        l = bufferL;
        for (auto &val : storage[i][j]->output) { val = 0.f; };
        for (size_t k = 0; k < topology(i + 1)->getFeatures(); k++) {
          storage[i][j]->output += storage[i + 1][l++]->errorInput;
        }
        if (topology(i + 1)->getFeatures() > 1) {
          storage[i][j]->output *= 1.f / (float) topology(i + 1)->getFeatures();
        }
      }
    }

    FloatMatrix bufferInput(input);
    for (size_t i = 0; i < topology(0)->getFeatures(); i++) {
      std::cout << "\n[input] [" << 0 << ", " << i << "]" << std::endl;
      layers[0][i]->computeBackward(bufferInput, *storage[0][i]);
    }
    for (auto &val : bufferInput) { val = 0.f; };
    for (size_t i = 0; i < topology(0)->getFeatures(); i++) {
      bufferInput += storage[0][i]->errorInput;
    }
    if (topology(0)->getFeatures() > 1) { bufferInput *= 1.f / (float) topology(0)->getFeatures(); }
    std::cout << bufferInput << std::endl;
  }

}   // namespace cnnet