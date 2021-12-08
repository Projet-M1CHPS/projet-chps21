#pragma once

#include <vector>

#include "BackpropStorage.hpp"
#include "Matrix.hpp"
#include "NeuralNetwork.hpp"
#include "TrainingMethod.hpp"

namespace nnet {

  template<typename T>
  class Optimizer {
  public:
    Optimizer() = default;
    ~Optimizer() = default;

    virtual void train(const math::Matrix<T> &inputs, const math::Matrix<T> &targets) = 0;
  };


  template<typename T>
  class MLPOptimizer : public Optimizer<T> {
  public:
    MLPOptimizer(NeuralNetwork<T> *const nn, TrainingMethod<T> *const tm)
        : neuralNetwork(nn), trainingMethod(tm), storage(nn->getWeights()) {
      layers.resize(nn->getWeights().size() + 1);
      layers_af.resize(nn->getWeights().size() + 1);
    };

    ~MLPOptimizer() = default;

    void train(const math::Matrix<T> &inputs, const math::Matrix<T> &targets) override {
      const size_t nbInput = inputs.getRows();
      const size_t nbTarget = targets.getRows();

      auto &weights = neuralNetwork->getWeights();

      if (nbInput != weights.front().getCols() || nbTarget != neuralNetwork->getOutputSize()) {
        throw std::runtime_error("Invalid number of inputs");
      }

      forward(inputs);
      backward(targets);
    }

  private:
    void forward(math::Matrix<T> const &inputs) {
      auto &weights = neuralNetwork->getWeights();
      auto &biases = neuralNetwork->getBiases();
      auto &activation_functions = neuralNetwork->getActivationFunctions();

      layers[0] = inputs;
      layers_af[0] = inputs;

      if (weights.empty()) return;

      math::Matrix<T> current_layer =
              math::Matrix<T>::matMatProdMatAdd(weights[0], inputs, biases[0]);
      layers[1] = current_layer;
      auto afunc = af::getAFFromType<T>(activation_functions[0]).first;
      std::transform(current_layer.cbegin(), current_layer.cend(), current_layer.begin(), afunc);
      layers_af[1] = current_layer;

      for (size_t k = 1; k < weights.size(); k++) {
        // C = W * C + B
        current_layer = math::Matrix<T>::matMatProdMatAdd(weights[k], current_layer, biases[k]);
        layers[k + 1] = current_layer;

        // Apply activation function on every element of the matrix
        afunc = af::getAFFromType<T>(activation_functions[k]).first;
        std::transform(current_layer.cbegin(), current_layer.cend(), current_layer.begin(), afunc);

        layers_af[k + 1] = current_layer;
      }
    }

    void backward(math::Matrix<T> const &target) {
      auto &weights = neuralNetwork->getWeights();
      auto &biases = neuralNetwork->getBiases();
      auto &activation_functions = neuralNetwork->getActivationFunctions();

      storage.current_error = layers_af[layers_af.size() - 1] - target;

      for (storage.index = weights.size() - 1; storage.index >= 0; storage.index--) {
        math::Matrix<T> derivative(layers[storage.index + 1]);
        auto dafunc = af::getAFFromType<T>(activation_functions[storage.index]).second;
        std::transform(derivative.cbegin(), derivative.cend(), derivative.begin(), dafunc);

        derivative.hadamardProd(storage.current_error);

        storage.current_error =
                math::Matrix<T>::mul(true, weights[storage.index], false, derivative);

        storage.gradient =
                math::Matrix<T>::mul(false, derivative, true, layers_af[storage.index], 1.0);

        trainingMethod->compute(storage);
      }
    }

  private:
    NeuralNetwork<T> *const neuralNetwork;
    TrainingMethod<T> *const trainingMethod;

    //
    std::vector<math::Matrix<T>> layers, layers_af;

    //
    BackpropStorage<T> storage;
  };

}   // namespace nnet