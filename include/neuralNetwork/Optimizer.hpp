#pragma once

#include <vector>

#include "BackpropStorage.hpp"
#include "Matrix.hpp"
#include "NeuralNetwork.hpp"
#include "OptimizationMethod.hpp"

namespace nnet {

  template<typename T>
  class Optimizer {
  public:
    Optimizer(NeuralNetwork<T> *const nn, OptimizationMethod<T> *const tm)
        : neural_network(nn), opti_meth(tm) {}

    Optimizer(const Optimizer<T> &other) = delete;
    Optimizer(Optimizer<T> &&other) noexcept = default;

    Optimizer<T> &operator=(const Optimizer<T> &other) = delete;
    Optimizer<T> &operator=(Optimizer<T> &&other) noexcept = default;

    NeuralNetwork<T> *getNeuralNetwork() const { return neural_network; }

    OptimizationMethod<T> *getOptimizationMethod() const { return opti_meth; }

    virtual ~Optimizer() = default;

  protected:
    NeuralNetwork<T> *const neural_network;
    OptimizationMethod<T> *const opti_meth;
  };


  template<typename T>
  class MLPStochOptimizer : public Optimizer<T> {
  public:
    MLPStochOptimizer(NeuralNetwork<T> *const nn, OptimizationMethod<T> *const tm)
        : Optimizer<T>(nn, tm), storage(this->neural_network->getWeights()) {
      layers.resize(nn->getWeights().size() + 1);
      layers_af.resize(nn->getWeights().size() + 1);
    };

    void train(const math::Matrix<T> &input, const math::Matrix<T> &target) {
      const size_t nbInput = input.getRows();
      const size_t nbTarget = target.getRows();

      auto &weights = this->neural_network->getWeights();

      if (nbInput != this->neural_network->getInputSize() ||
          nbTarget != this->neural_network->getOutputSize()) {
        throw std::runtime_error("Invalid number of inputs");
      }

      forward(input);
      storage.getError() = layers_af[layers_af.size() - 1] - target;
      backward(target);
    }

    template<typename input_iterator, typename target_iterator>
    void train(const input_iterator begin, const input_iterator end,
               const target_iterator target_begin) {
      for (auto it = begin; it != end; ++it, ++target_begin) { train(*it, *target_begin); }
    }


  private:
    void forward(math::Matrix<T> const &inputs) {
      auto &weights = this->neural_network->getWeights();
      auto &biases = this->neural_network->getBiases();
      auto &activation_functions = this->neural_network->getActivationFunctions();

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
      auto &weights = this->neural_network->getWeights();
      auto &biases = this->neural_network->getBiases();
      auto &activation_functions = this->neural_network->getActivationFunctions();

      // storage.current_error = layers_af[layers_af.size() - 1] - target;

      for (long i = weights.size() - 1; i >= 0; i--) {
        storage.setIndex(i);

        math::Matrix<T> derivative(layers[i + 1]);
        auto dafunc = af::getAFFromType<T>(activation_functions[i]).second;
        std::transform(derivative.cbegin(), derivative.cend(), derivative.begin(), dafunc);

        derivative.hadamardProd(storage.getError());

        storage.getError() = math::Matrix<T>::mul(true, weights[i], false, derivative);

        storage.getGradient() = math::Matrix<T>::mul(false, derivative, true, layers_af[i], 1.0);

        this->opti_meth->compute(storage);
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
    MLPBatchOptimizer(NeuralNetwork<T> *const nn, OptimizationMethod<T> *const tm)
        : Optimizer<T>(nn, tm), storage(this->neural_network->getWeights()) {
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

    void train(const std::vector<math::Matrix<T>> &inputs,
               const std::vector<math::Matrix<T>> &targets) {
      train(inputs.begin(), inputs.end(), targets.begin());
    }

    template<typename input_iterator, typename target_iterator>
    void train(const input_iterator begin, const input_iterator end,
               const target_iterator targets_beg) {
      size_t n = std::distance(begin, end);

      auto mat_reset = [](math::Matrix<T> &m) { m.fill(0); };
      std::for_each(avg_gradients.begin(), avg_gradients.end(), mat_reset);
      std::for_each(avg_errors.begin(), avg_errors.end(), mat_reset);

      long i = 0;
      auto it_target = targets_beg;

      for (auto it = begin; it != end; it++, it_target++, i++) {
        forward(*it);
        storage.getError() = layers_af[layers_af.size() - 1] - *it_target;
        computeGradient();
      }

      for (auto &it : avg_gradients) { it *= ((T) 1.0 / n); }

      for (i = this->neural_network->getWeights().size() - 1; i >= 0; i--) {
        storage.getGradient() = avg_gradients[i];
        storage.getError() = avg_errors[i];
        storage.setIndex(i);
        this->opti_meth->compute(storage);
      }
    }

  private:
    void forward(math::Matrix<T> const &inputs) {
      auto &weights = this->neural_network->getWeights();
      auto &biases = this->neural_network->getBiases();
      auto &activation_functions = this->neural_network->getActivationFunctions();

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
      auto &weights = this->neural_network->getWeights();
      auto &biases = this->neural_network->getBiases();
      auto &activation_functions = this->neural_network->getActivationFunctions();

      // Avoid the loop index underflowing back to +inf
      if (weights.empty()) return;

      for (long i = weights.size() - 1; i >= 0; i--) {
        storage.setIndex(i);

        math::Matrix<T> derivative(layers[i + 1]);
        auto dafunc = af::getAFFromType<T>(activation_functions[i]).second;
        std::transform(derivative.cbegin(), derivative.cend(), derivative.begin(), dafunc);

        derivative.hadamardProd(storage.getError());

        storage.getError() = math::Matrix<T>::mul(true, weights[i], false, derivative);

        storage.getGradient() = math::Matrix<T>::mul(false, derivative, true, layers_af[i], 1.0);

        avg_gradients[i] += storage.getGradient();
      }
    }

    void backward(math::Matrix<T> const &target) {
      auto &weights = this->neural_network->getWeights();
      auto &biases = this->neural_network->getBiases();
      auto &activation_functions = this->neural_network->getActivationFunctions();

      // Avoid the loop index underflowing back to +inf
      if (weights.empty()) return;

      for (long i = weights.size() - 1; i >= 0; i--) {
        math::Matrix<T> derivative(layers[storage.index + 1]);
        auto dafunc = af::getAFFromType<T>(activation_functions[storage.index]).second;
        std::transform(derivative.cbegin(), derivative.cend(), derivative.begin(), dafunc);

        derivative.hadamardProd(storage.current_error);

        storage.getError() = math::Matrix<T>::mul(true, weights[storage.index], false, derivative);

        storage.getGradient() =
                math::Matrix<T>::mul(false, derivative, true, layers_af[storage.index], 1.0);

        this->opti_meth->compute(storage);
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
