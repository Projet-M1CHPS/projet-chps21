#include "CNNModelOptimizer.hpp"


namespace cnnet {

  CNNModelOptimizer::CNNModelOptimizer(
          CNNModel &model)   //, std::shared_ptr<OptimizationMethod> tm)
      : nn_cnn(&model.getCnn()), flatten(&model.getFlatten()) {   //, opti_meth(tm) {
    // mlpOpti = MLPModelOptimizer(model.getMlp(), tm);

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

  CNNModelStochOptimizer::CNNModelStochOptimizer(CNNModel &model) : CNNModelOptimizer(model) {}

  void CNNModelStochOptimizer::train(const math::FloatMatrix &input,
                                     const math::FloatMatrix &target) {
    forward(input);

    // FloatMatrix errorFlatten = mlpOpti.train(flatten, target);


    for (size_t i = 0; i < storage.size(); i++) {
      for (size_t j = 0; j < storage[i].size(); j++) {
        std::cout << storage[i][j]->output << std::endl;
      }
      std::cout << std::endl;
    }
    //std::cout << "flatten \n" << *flatten << std::endl;

    FloatMatrix errorFlatten = FloatMatrix(flatten->getRows(), flatten->getCols());
    for (auto &i : errorFlatten) { i = 1.f; }


    backward(input, errorFlatten);
  }

  void CNNModelStochOptimizer::optimize(const std::vector<math::FloatMatrix> &inputs,
                                        const std::vector<math::FloatMatrix> &targets) {
    if (inputs.size() != targets.size()) {
      throw std::runtime_error("Invalid number of inputs or targets");
    }
    for (size_t i = 0; i < inputs.size(); ++i) { train(inputs[i], targets[i]); }
  }

  void CNNModelStochOptimizer::forward(math::FloatMatrix const &input) {
    std::cout << "forward" << std::endl;
    auto &layers = nn_cnn->getLayers();
    auto &topology = nn_cnn->getTopology();

    for (size_t i = 0; i < topology(0)->getFeatures(); i++)
      layers[0][i]->compute(input, storage[0][i]->output);

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
        (*flatten)(index, 0) = val;
        index++;
      }
    }
  }

  void CNNModelStochOptimizer::backward(math::FloatMatrix const &input,
                                        math::FloatMatrix const &errorFlatten) {
    std::cout << "backward" << std::endl;
    // Convert flatten to matrix
    size_t index = 0;
    for (auto &storageElement : storage.back()) {
      for (auto &val : storageElement->output) {
        val = errorFlatten(index, 0);
        index++;
      }
    }

    auto &layers = nn_cnn->getLayers();
    auto &topology = nn_cnn->getTopology();

    for (long i = storage.size() - 2; i >= 0; i--) {
      size_t l = 0;
      for (size_t j = 0; j < storage[i].size(); j++) {
        for(size_t k = 0; k < topology(i + 1)->getFeatures(); k++) {
          std::cout << "\n[" << i << ", " << j << "] [" << i + 1 << ", " << l << "]" << std::endl;
          std::cout << storage[i+1][l]->output << std::endl;
          layers[i + 1][j]->computeBackward(storage[i][j]->output, *storage[i + 1][l++]);
          std::cout << storage[i][j]->output << std::endl;
        }
      }
    }
  }

}   // namespace cnnet