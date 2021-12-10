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
    Optimizer(NeuralNetwork<T> *const nn, TrainingMethod<T> *const tm)
        : neuralNetwork(nn), trainingMethod(tm) {}
    virtual ~Optimizer() = default;

    virtual void train(const math::Matrix<T> &inputs, const math::Matrix<T> &targets) = 0;
    virtual void train(const std::vector<math::Matrix<T>> &inputs,
                       const std::vector<math::Matrix<T>> &targets) = 0;

  protected:
    NeuralNetwork<T> *const neuralNetwork;
    TrainingMethod<T> *const trainingMethod;
  };


  template<typename T>
  class MLPStochOptimizer : public Optimizer<T> {
  public:
    MLPStochOptimizer(NeuralNetwork<T> *const nn, TrainingMethod<T> *const tm)
        : Optimizer<T>(nn, tm), storage(this->neuralNetwork->getWeights()) {
      layers.resize(nn->getWeights().size() + 1);
      layers_af.resize(nn->getWeights().size() + 1);
    };

    ~MLPStochOptimizer() = default;

    void train(const math::Matrix<T> &input, const math::Matrix<T> &target) override {
      const size_t nbInput = input.getRows();
      const size_t nbTarget = target.getRows();

      auto &weights = this->neuralNetwork->getWeights();

      if (nbInput != this->neuralNetwork->getInputSize() ||
          nbTarget != this->neuralNetwork->getOutputSize()) {
        throw std::runtime_error("Invalid number of inputs");
      }

      forward(input);
      storage.current_error = layers_af[layers_af.size() - 1] - target;
      backward(target);
    }

    void train(const std::vector<math::Matrix<T>> &inputs,
               const std::vector<math::Matrix<T>> &targets) {}


  private:
    void forward(math::Matrix<T> const &inputs) {
      auto &weights = this->neuralNetwork->getWeights();
      auto &biases = this->neuralNetwork->getBiases();
      auto &activation_functions = this->neuralNetwork->getActivationFunctions();

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
      auto &weights = this->neuralNetwork->getWeights();
      auto &biases = this->neuralNetwork->getBiases();
      auto &activation_functions = this->neuralNetwork->getActivationFunctions();

      // storage.current_error = layers_af[layers_af.size() - 1] - target;

      std::cout << "gradient" << std::endl;
      for (storage.index = weights.size() - 1; storage.index >= 0; storage.index--) {
        math::Matrix<T> derivative(layers[storage.index + 1]);
        auto dafunc = af::getAFFromType<T>(activation_functions[storage.index]).second;
        std::transform(derivative.cbegin(), derivative.cend(), derivative.begin(), dafunc);

        derivative.hadamardProd(storage.current_error);

        storage.current_error =
                math::Matrix<T>::mul(true, weights[storage.index], false, derivative);

        storage.gradient =
                math::Matrix<T>::mul(false, derivative, true, layers_af[storage.index], 1.0);

        std::cout << storage.gradient << std::endl;

        // this->trainingMethod->compute(storage);
      }
    }

  private:
    //
    std::vector<math::Matrix<T>> layers, layers_af;

    //
    BackpropStorage<T> storage;
  };


  template<typename T>
  class MLPBatchOptimizer : public Optimizer<T> {
  public:
    MLPBatchOptimizer(NeuralNetwork<T> *const nn, TrainingMethod<T> *const tm)
        : Optimizer<T>(nn, tm), storage(this->neuralNetwork->getWeights()) {
      layers.resize(nn->getWeights().size() + 1);
      layers_af.resize(nn->getWeights().size() + 1);

      const auto &topology = nn->getTopology();

      for (size_t i = 0; i < nn->getWeights().size(); i++) {
        avg_gradients.push_back(math::Matrix<T>(topology[i + 1], topology[i]));
        avg_errors.push_back(math::Matrix<T>(topology[i + 1], 1));

        avg_gradients[i].fill(0.0);
        avg_errors[i].fill(0.0);
      }
    };

    ~MLPBatchOptimizer() = default;

    void train(const math::Matrix<T> &input, const math::Matrix<T> &target) override {
      const size_t nbInput = input.getRows();
      const size_t nbTarget = target.getRows();

      auto &weights = this->neuralNetwork->getWeights();

      if (nbInput != this->neuralNetwork->getInputSize() ||
          nbTarget != this->neuralNetwork->getOutputSize()) {
        throw std::runtime_error("Invalid number of inputs");
      }

      forward(input);
      storage.current_error = layers_af[layers_af.size() - 1] - target;
      backward(target);
    }

    void train(const std::vector<math::Matrix<T>> &inputs,
               const std::vector<math::Matrix<T>> &targets) {
      for (size_t i = 0; i < avg_gradients.size(); i++) {
        avg_gradients[i].fill(0.0);
        avg_errors[i].fill(0.0);
      }

      for (size_t i = 0; i < inputs.size(); i++) {
        forward(inputs[i]);

        storage.current_error = layers_af[layers_af.size() - 1] - targets[i];

        computeGradient();
      }

      // std::cout << "Gradients computed" << std::endl;

      for (auto &i : avg_gradients) {
        i *= (T) 1.0 / inputs.size();
        // std::cout << i << std::endl;
      }


      for (storage.index = this->neuralNetwork->getWeights().size() - 1; storage.index >= 0;
           storage.index--) {
        storage.gradient = avg_gradients[storage.index];
        storage.current_error = avg_errors[storage.index];
        this->trainingMethod->compute(storage);
      }
    }

  private:
    void forward(math::Matrix<T> const &inputs) {
      auto &weights = this->neuralNetwork->getWeights();
      auto &biases = this->neuralNetwork->getBiases();
      auto &activation_functions = this->neuralNetwork->getActivationFunctions();

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

    void computeGradient() {
      auto &weights = this->neuralNetwork->getWeights();
      auto &biases = this->neuralNetwork->getBiases();
      auto &activation_functions = this->neuralNetwork->getActivationFunctions();

      for (storage.index = weights.size() - 1; storage.index >= 0; storage.index--) {
        math::Matrix<T> derivative(layers[storage.index + 1]);
        auto dafunc = af::getAFFromType<T>(activation_functions[storage.index]).second;
        std::transform(derivative.cbegin(), derivative.cend(), derivative.begin(), dafunc);

        derivative.hadamardProd(storage.current_error);

        // avg_errors[storage.index] += storage.current_error;

        storage.current_error =
                math::Matrix<T>::mul(true, weights[storage.index], false, derivative);

        storage.gradient =
                math::Matrix<T>::mul(false, derivative, true, layers_af[storage.index], 1.0);

        // std::cout << "rapide\n" << storage.gradient << std::endl;

        avg_gradients[storage.index] += storage.gradient;
      }
    }

    void backward(math::Matrix<T> const &target) {
      auto &weights = this->neuralNetwork->getWeights();
      auto &biases = this->neuralNetwork->getBiases();
      auto &activation_functions = this->neuralNetwork->getActivationFunctions();


      for (storage.index = weights.size() - 1; storage.index >= 0; storage.index--) {
        math::Matrix<T> derivative(layers[storage.index + 1]);
        auto dafunc = af::getAFFromType<T>(activation_functions[storage.index]).second;
        std::transform(derivative.cbegin(), derivative.cend(), derivative.begin(), dafunc);

        derivative.hadamardProd(storage.current_error);

        storage.current_error =
                math::Matrix<T>::mul(true, weights[storage.index], false, derivative);

        storage.gradient =
                math::Matrix<T>::mul(false, derivative, true, layers_af[storage.index], 1.0);

        this->trainingMethod->compute(storage);
      }
    }

  private:
    //
    std::vector<math::Matrix<T>> layers, layers_af;

    //
    BackpropStorage<T> storage;

    std::vector<math::Matrix<T>> avg_errors;
    std::vector<math::Matrix<T>> avg_gradients;
  };


}   // namespace nnet
