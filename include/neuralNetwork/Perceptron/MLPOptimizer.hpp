#pragma once

#include "MLPModel.hpp"
#include "MLPerceptron.hpp"
#include "OptimizationMethod.hpp"
#include "neuralNetwork/ModelOptimizer.hpp"
#include <iostream>

namespace nnet {

  template<typename real = float>
  class MLPModelOptimizer : public ModelOptimizer<real> {
  public:
    MLPModelOptimizer(MLPModel<real> *const model, OptimizationMethod<real> *const tm)
        : neural_network(&model->getPerceptron()), opti_meth(tm) {}

    MLPModelOptimizer(const MLPModelOptimizer<real> &other) = delete;
    MLPModelOptimizer(MLPModelOptimizer<real> &&other) noexcept = default;

    MLPModelOptimizer<real> &operator=(const MLPModelOptimizer<real> &other) = delete;
    MLPModelOptimizer<real> &operator=(MLPModelOptimizer<real> &&other) noexcept = default;

    MLPerceptron<real> *getNeuralNetwork() const { return neural_network; }
    OptimizationMethod<real> *getOptimizationMethod() const { return opti_meth; }

    virtual ~MLPModelOptimizer() = default;

    virtual void optimize(const std::vector<math::Matrix<real>> &inputs,
                          const std::vector<math::Matrix<real>> &targets) = 0;

    virtual void update() { opti_meth->update(); }

  protected:
    MLPerceptron<real> *const neural_network;
    OptimizationMethod<real> *const opti_meth;
  };


  template<typename real = float>
  class MLPModelStochOptimizer final : public MLPModelOptimizer<real> {
  public:
    MLPModelStochOptimizer(MLPModel<real> *const model, OptimizationMethod<real> *const tm)
        : MLPModelOptimizer<real>(model, tm), storage(this->neural_network->getWeights()) {
      layers.resize(this->neural_network->getWeights().size() + 1);
      layers_af.resize(this->neural_network->getWeights().size() + 1);
    };


    void train(const math::Matrix<real> &input, const math::Matrix<real> &target) {
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


    void optimize(const std::vector<math::Matrix<real>> &inputs,
                  const std::vector<math::Matrix<real>> &targets) override {
      if (inputs.size() != targets.size()) { throw std::runtime_error("Invalid number of inputs"); }
      for (size_t i = 0; i < inputs.size(); ++i) { train(inputs[i], targets[i]); }
    }

  private:
    void forward(math::Matrix<real> const &inputs) {
      auto &weights = this->neural_network->getWeights();
      auto &biases = this->neural_network->getBiases();
      auto &activation_functions = this->neural_network->getActivationFunctions();

      layers[0] = inputs;
      layers_af[0] = inputs;

      if (weights.empty()) return;

      math::Matrix<real> current_layer =
              math::Matrix<real>::matMatProdMatAdd(weights[0], inputs, biases[0]);
      layers[1] = current_layer;
      auto afunc = af::getAFFromType<real>(activation_functions[0]).first;
      std::transform(current_layer.cbegin(), current_layer.cend(), current_layer.begin(), afunc);
      layers_af[1] = current_layer;

      for (size_t k = 1; k < weights.size(); k++) {
        // C = W * C + B
        current_layer = math::Matrix<real>::matMatProdMatAdd(weights[k], current_layer, biases[k]);
        layers[k + 1] = current_layer;

        // Apply activation function on every element of the matrix
        afunc = af::getAFFromType<real>(activation_functions[k]).first;
        std::transform(current_layer.cbegin(), current_layer.cend(), current_layer.begin(), afunc);

        layers_af[k + 1] = current_layer;
      }
    }

    void backward(math::Matrix<real> const &target) {
      auto &weights = this->neural_network->getWeights();
      auto &biases = this->neural_network->getBiases();
      auto &activation_functions = this->neural_network->getActivationFunctions();

      // storage.current_error = layers_af[layers_af.size() - 1] - target;

      for (long i = weights.size() - 1; i >= 0; i--) {
        storage.setIndex(i);

        math::Matrix<real> derivative(layers[i + 1]);
        auto dafunc = af::getAFFromType<real>(activation_functions[i]).second;
        std::transform(derivative.cbegin(), derivative.cend(), derivative.begin(), dafunc);

        derivative.hadamardProd(storage.getError());

        storage.getError() = math::Matrix<real>::mul(true, weights[i], false, derivative);

        storage.getGradient() = math::Matrix<real>::mul(false, derivative, true, layers_af[i], 1.0);

        this->opti_meth->compute(storage);
      }
    }

  private:
    //
    std::vector<math::Matrix<real>> layers, layers_af;

    //
    BackpropStorage<real> storage;
  };


  template<typename real = float>
  class MLPBatchOptimizer final : public MLPModelOptimizer<real> {
  public:
    MLPBatchOptimizer(MLPerceptron<real> *const nn, OptimizationMethod<real> *const tm)
        : MLPModelOptimizer<real>(nn, tm), storage(this->neural_network->getWeights()) {
      layers.resize(nn->getWeights().size() + 1);
      layers_af.resize(nn->getWeights().size() + 1);

      const auto &topology = nn->getTopology();

      for (size_t i = 0; i < nn->getWeights().size(); i++) {
        avg_gradients.push_back(math::Matrix<real>(topology[i + 1], topology[i]));
        avg_errors.push_back(math::Matrix<real>(topology[i + 1], 1));

        avg_gradients[i].fill(0.0);
        avg_errors[i].fill(0.0);
      }
    };

    ~MLPBatchOptimizer() = default;

    void optimize(const std::vector<math::Matrix<real>> &inputs,
                  const std::vector<math::Matrix<real>> &targets) override {
      size_t n = inputs.size();

      auto mat_reset = [](math::Matrix<real> &m) { m.fill(0); };
      std::for_each(avg_gradients.begin(), avg_gradients.end(), mat_reset);
      std::for_each(avg_errors.begin(), avg_errors.end(), mat_reset);

      for (long i = 0; i < n; i++) {
        forward(inputs[i]);
        storage.getError() = layers_af[layers_af.size() - 1] - targets[i];
        computeGradient();
      }

      for (auto &it : avg_gradients) { it *= ((real) 1.0 / n); }

      for (long i = this->neural_network->getWeights().size() - 1; i >= 0; i--) {
        storage.setIndex(i);
        storage.getGradient() = avg_gradients[i];
        storage.getError() = avg_errors[i];
        this->opti_meth->compute(storage);
      }
    }

  private:
    void forward(math::Matrix<real> const &inputs) {
      auto &weights = this->neural_network->getWeights();
      auto &biases = this->neural_network->getBiases();
      auto &activation_functions = this->neural_network->getActivationFunctions();

      layers[0] = inputs;
      layers_af[0] = inputs;

      if (weights.empty()) return;

      math::Matrix<real> current_layer =
              math::Matrix<real>::matMatProdMatAdd(weights[0], inputs, biases[0]);
      layers[1] = current_layer;
      auto afunc = af::getAFFromType<real>(activation_functions[0]).first;
      std::transform(current_layer.cbegin(), current_layer.cend(), current_layer.begin(), afunc);
      layers_af[1] = current_layer;

      for (size_t k = 1; k < weights.size(); k++) {
        // C = W * C + B
        current_layer = math::Matrix<real>::matMatProdMatAdd(weights[k], current_layer, biases[k]);
        layers[k + 1] = current_layer;

        // Apply activation function on every element of the matrix
        afunc = af::getAFFromType<real>(activation_functions[k]).first;
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

        math::Matrix<real> derivative(layers[i + 1]);
        auto dafunc = af::getAFFromType<real>(activation_functions[i]).second;
        std::transform(derivative.cbegin(), derivative.cend(), derivative.begin(), dafunc);

        derivative.hadamardProd(storage.getError());

        storage.getError() = math::Matrix<real>::mul(true, weights[i], false, derivative);

        storage.getGradient() = math::Matrix<real>::mul(false, derivative, true, layers_af[i], 1.0);

        avg_gradients[i] += storage.getGradient();
      }
    }

    void backward(math::Matrix<real> const &target) {
      auto &weights = this->neural_network->getWeights();
      auto &biases = this->neural_network->getBiases();
      auto &activation_functions = this->neural_network->getActivationFunctions();

      // Avoid the loop index underflowing back to +inf
      if (weights.empty()) return;

      for (long i = weights.size() - 1; i >= 0; i--) {
        storage.setIndex(i);
        math::Matrix<real> derivative(layers[storage.index + 1]);
        auto dafunc = af::getAFFromType<real>(activation_functions[storage.index]).second;
        std::transform(derivative.cbegin(), derivative.cend(), derivative.begin(), dafunc);

        derivative.hadamardProd(storage.current_error);

        storage.getError() =
                math::Matrix<real>::mul(true, weights[storage.index], false, derivative);

        storage.getGradient() =
                math::Matrix<real>::mul(false, derivative, true, layers_af[storage.index], 1.0);

        this->opti_meth->compute(storage);
      }
    }

  private:
    //
    std::vector<math::Matrix<real>> layers, layers_af;

    //
    BackpropStorage<real> storage;

    std::vector<math::Matrix<real>> avg_errors;
    std::vector<math::Matrix<real>> avg_gradients;
  };

}   // namespace nnet