#pragma once

#include "ActivationFunction.hpp"
#include "Utils.hpp"
#include "math/Matrix.hpp"
#include "math/clFMatrix.hpp"
#include "math/clFTensor.hpp"
#include "openclUtils/clWrapper.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

namespace nnet {

  /**
   * @brief Describes the layers of a multilayer perceptron.
   */
  class MLPTopology {
    friend std::ostream &operator<<(std::ostream &os, const MLPTopology &topology);

  public:
    /**
     * @brief Builds a topology from a space separated string
     * @param str
     * @return
     */
    static MLPTopology fromString(const std::string &str);

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

    /**
     * @brief Insert a layer to the front of the topology
     * @param i
     */
    void pushFront(size_t i) { layers.insert(layers.begin(), i); }

    /**
     * @brief Append a layer to the back of the topology
     * @param i
     */
    void pushBack(size_t i) { layers.push_back(i); }

    /**
     * @brief Returns true if the topology is empty
     * @return
     */
    [[nodiscard]] bool empty() const { return layers.empty(); }

    /**
     * @brief Returns the number of layers in the topology
     * @return
     */
    [[nodiscard]] size_t size() const { return layers.size(); }

    using iterator = std::vector<size_t>::iterator;
    using const_iterator = std::vector<size_t>::const_iterator;

    iterator begin() { return layers.begin(); }
    iterator end() { return layers.end(); }

    [[nodiscard]] const_iterator begin() const { return layers.begin(); }
    [[nodiscard]] const_iterator end() const { return layers.end(); }

  private:
    std::vector<size_t> layers;
  };

  /**
   * @brief A multilayer Perceptron neural network
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
     * @return
     */
    math::clFMatrix predict(cl::CommandQueue &queue, math::clFMatrix const &input) const;

    math::clFMatrix predict(math::clFMatrix const &input) const;


    /**
     * @brief Predict the output of the neural network on the given input
     * Uses the next available queue
     * @param input
     * @return
     */
    math::clFTensor predict(cl::CommandQueue &queue, math::clFTensor const &input) const;
    math::clFTensor predict(math::clFTensor const &input) const;


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