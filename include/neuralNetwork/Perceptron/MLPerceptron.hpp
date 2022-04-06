#pragma once

#include "ActivationFunction.hpp"
#include "Matrix.hpp"
#include "Utils.hpp"
#include "clUtils/clFMatrix.hpp"
#include "clUtils/clFTensor.hpp"
#include "clUtils/clWrapper.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

namespace nnet {

  /**
   * @brief Wrapper around an std::vector used to describe the layers of an MLP
   */
  class MLPTopology {
    friend std::ostream &operator<<(std::ostream &os, const MLPTopology &topology);

  public:
    MLPTopology() = default;
    MLPTopology(std::initializer_list<size_t> list) : layers(list) {}

    explicit MLPTopology(std::vector<size_t> sizes) : layers(std::move(sizes)) {}
    explicit MLPTopology(std::vector<size_t> &&sizes) : layers(std::move(sizes)) {}

    size_t &operator[](size_t i) {
      if (i > layers.size()) { throw std::out_of_range("Index out of range"); }
      return layers[i];
    }

    size_t const &operator[](size_t i) const {
      if (i > layers.size()) { throw std::out_of_range("Index out of range"); }
      return layers[i];
    }

    [[nodiscard]] size_t getInputSize() const { return layers.empty() ? 0 : layers.front(); }
    void setInputSize(size_t i) {
      if (layers.empty()) {
        layers.push_back(i);
      } else {
        layers[0] = i;
      }
    }

    [[nodiscard]] size_t getOutputSize() const { return layers.empty() ? 0 : layers.back(); }
    void setOutputSize(size_t i) {
      if (layers.empty()) {
        layers.push_back(i);
      } else {
        layers[layers.size() - 1] = i;
      }
    }

    void pushFront(size_t i) { layers.insert(layers.begin(), i); }
    void pushBack(size_t i) { layers.push_back(i); }

    [[nodiscard]] bool empty() const { return layers.empty(); }
    [[nodiscard]] size_t size() const { return layers.size(); }

    using iterator = std::vector<size_t>::iterator;
    using const_iterator = std::vector<size_t>::const_iterator;

    iterator begin() { return layers.begin(); }
    iterator end() { return layers.end(); }

    static MLPTopology fromString(const std::string &str);
    [[nodiscard]] const_iterator begin() const { return layers.begin(); }
    [[nodiscard]] const_iterator end() const { return layers.end(); }

  private:
    std::vector<size_t> layers;
  };

  /**
   * @brief A neural network that supports most fp precision as template
   * parameters
   *
   */
  class MLPerceptron final {
  public:
    /**
     * @brief Construct a new Neural Network object with no layer
     *
     */
    explicit MLPerceptron(const MLPTopology &topology = {});

    MLPerceptron(const MLPerceptron &other) { *this = other; }
    MLPerceptron &operator=(const MLPerceptron &);

    MLPerceptron(MLPerceptron &&other) noexcept = default;
    MLPerceptron &operator=(MLPerceptron &&) noexcept = default;

    /**
     * @brief Predict the output of the neural network on the given input
     * Uses the next available queue
     * @param input
     * @param qhandler
     * @return
     */
    math::clFMatrix predict(math::clFMatrix const &input) const;

    [[nodiscard]] MLPTopology const &getTopology() const { return topology; }

    /**
     * @brief Take a vector of sizes correspondig to the number of neurons
     * in each layer and build the network accordingly. Note that weights are not
     * initialized after this.

     * @param topology
     */
    void setTopology(MLPTopology const &topology);
    void setActivationFunction(af::ActivationFunctionType type) {
      for (auto &activation_function : activation_functions) { activation_function = type; }
    }

    void setActivationFunction(af::ActivationFunctionType af, size_t layer) {
      if (layer >= weights.size()) { throw std::invalid_argument("Invalid layer"); }

      activation_functions[layer] = af;
    }

    [[nodiscard]] const std::vector<af::ActivationFunctionType> &getActivationFunctions() const {
      return activation_functions;
    }

    /**
     * @brief Randomizes the weights and biases of the network
     *
     */
    void randomizeWeight();

    [[nodiscard]] std::vector<math::clFMatrix> &getWeights() { return weights; }
    [[nodiscard]] const std::vector<math::clFMatrix> &getWeights() const { return weights; }

    [[nodiscard]] std::vector<math::clFMatrix> &getBiases() { return biases; }
    [[nodiscard]] const std::vector<math::clFMatrix> &getBiases() const { return biases; }

  private:
    MLPTopology topology;

    std::vector<math::clFMatrix> weights;
    std::vector<math::clFMatrix> biases;

    // We want every layer to have its own activation function
    std::vector<af::ActivationFunctionType> activation_functions;
  };

  // std::ostream& operator<<(std::ostream& os, const Pair<T, U>& p)
  std::ostream &operator<<(std::ostream &os, const MLPerceptron &nn);

}   // namespace nnet