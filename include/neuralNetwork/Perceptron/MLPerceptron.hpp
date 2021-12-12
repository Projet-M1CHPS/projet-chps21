#pragma once

#include "Matrix.hpp"
#include "Utils.hpp"
#include "neuralNetwork/ActivationFunction.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

namespace nnet {

  /**
   * @brief Enum of the supported floating precision
   *
   * Used for serialization
   *
   */
  enum class FloatingPrecision {
    float32,
    float64,
  };

  class MLPTopology {
    friend std::ostream &operator<<(std::ostream &os, const MLPTopology &topology);

  public:
    MLPTopology() = default;
    MLPTopology(std::initializer_list<size_t> list) : layers(list) {}
    explicit MLPTopology(std::vector<size_t> sizes) : layers(std::move(sizes)) {}


    size_t &operator[](size_t i) {
      if (i > layers.size()) { throw std::out_of_range("Index out of range"); }
      return layers[i];
    }

    size_t const &operator[](size_t i) const {
      if (i > layers.size()) { throw std::out_of_range("Index out of range"); }
      return layers[i];
    }

    [[nodiscard]] size_t getInputSize() const { return layers.front(); }
    void setInputSize(size_t i) { layers.front() = i; }

    [[nodiscard]] size_t getOutputSize() const { return layers.back(); }
    void setOutputSize(size_t i) { layers.back() = i; }
    void push_back(size_t i) { layers.push_back(i); }

    [[nodiscard]] bool empty() const { return layers.empty(); }
    [[nodiscard]] size_t size() const { return layers.size(); }

    using iterator = std::vector<size_t>::iterator;
    using const_iterator = std::vector<size_t>::const_iterator;

    iterator begin() { return layers.begin(); }
    iterator end() { return layers.end(); }

    static MLPTopology fromString(const std::string &str) {
      std::vector<size_t> layers;
      std::stringstream ss(str);
      std::string token;
      while (std::getline(ss, token, ',')) { layers.push_back(std::stoi(token)); }
      return MLPTopology(layers);
    }

    [[nodiscard]] const_iterator begin() const { return layers.begin(); }
    [[nodiscard]] const_iterator end() const { return layers.end(); }

  private:
    std::vector<size_t> layers;
  };

  /**
   * @brief Interface for a neural network
   *
   */
  class MLPBase {
  public:
    MLPBase() = default;

    virtual ~MLPBase() = default;

    virtual void setTopology(MLPTopology const &topology) = 0;
    [[nodiscard]] MLPTopology const &getTopology() const { return topology; }

    virtual void randomizeWeight() = 0;

  protected:
    MLPTopology topology;
  };

  /**
   * @brief A neural network that supports most fp precision as template
   * parameters
   *
   * @tparam real
   */
  template<typename real = float, typename = std::enable_if<std::is_floating_point_v<real>>>
  class MLPerceptron final : public MLPBase {
  public:
    using value_type = real;

    /**
     * @brief Construct a new Neural Network object with no layer
     *
     */
    MLPerceptron() = default;

    MLPerceptron(const MLPerceptron &other) : MLPBase() { *this = other; }

    MLPerceptron(MLPerceptron &&other) noexcept : MLPBase() { *this = std::move(other); }

    MLPerceptron &operator=(const MLPerceptron &) = default;

    MLPerceptron &operator=(MLPerceptron &&other) noexcept {
      weights = std::move(other.weights);
      biases = std::move(other.biases);
      activation_functions = std::move(other.activation_functions);

      return *this;
    }

    /** @brief Runs the neural network on the inputs
     * The outputs are returned as a matrix of reals
     *
     * @tparam iterator
     * @param begin
     * @param end
     * @return
     */
    math::Matrix<real> predict(math::Matrix<real> const &input) const {
      const size_t nbInput = input.getRows();

      if (nbInput != weights.front().getCols()) {
        throw std::runtime_error("Invalid number of input");
      }

      auto current_layer = math::Matrix<real>::matMatProdMatAdd(weights[0], input, biases[0]);
      auto afunc = af::getAFFromType<real>(activation_functions[0]).first;
      std::transform(current_layer.cbegin(), current_layer.cend(), current_layer.begin(), afunc);

      for (size_t i = 1; i < weights.size(); i++) {
        current_layer = math::Matrix<real>::matMatProdMatAdd(weights[i], current_layer, biases[i]);

        // Apply activation function on every element of the matrix
        afunc = af::getAFFromType<real>(activation_functions[i]).first;
        std::transform(current_layer.cbegin(), current_layer.cend(), current_layer.begin(), afunc);
      }
      return current_layer;
    }

    /**
     * @brief Take a vector of sizes correspondig to the number of neurons
     * in each layer and build the network accordingly. Note that weights are not
     * initialized after this.
     *
     * @param layers
     */
    void setTopology(MLPTopology const &topology) override {
      if (topology.empty()) return;
      if (topology.size() < 2) { throw std::runtime_error("Requires atleast 2 layers"); }

      weights.clear();
      biases.clear();
      for (size_t i = 0; i < topology.size() - 1; i++) {
        // Create a matrix of size (layers[i + 1] x layers[i])
        // So that each weight matrix can be multiplied by the previous layer
        weights.push_back(math::Matrix<real>(topology[i + 1], topology[i]));
        biases.push_back(math::Matrix<real>(topology[i + 1], 1));
        activation_functions.push_back(af::ActivationFunctionType::sigmoid);
      }
      this->topology = topology;
    }

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
     * @param seed
     */
    void randomizeWeight() override {
      for (auto &layer : weights) {
        double x = std::sqrt(2.0 / (double) layer.getRows());
        math::randomize<real>(layer, -x, x);
      }

      for (auto &layer : biases) {
        double x = std::sqrt(2.0 / (double) layer.getRows());
        math::randomize<real>(layer, -x, x);
      }
    }

    [[nodiscard]] std::vector<math::Matrix<real>> &getWeights() { return weights; }

    [[nodiscard]] const std::vector<math::Matrix<real>> &getWeights() const { return weights; }

    [[nodiscard]] std::vector<math::Matrix<real>> &getBiases() { return biases; }

    [[nodiscard]] const std::vector<math::Matrix<real>> &getBiases() const { return biases; }


  private:
    std::vector<math::Matrix<real>> weights;
    std::vector<math::Matrix<real>> biases;

    // We want every layer to have its own activation function
    std::vector<af::ActivationFunctionType> activation_functions;
  };

  // std::ostream& operator<<(std::ostream& os, const Pair<T, U>& p)
  template<typename T>
  std::ostream &operator<<(std::ostream &os, const MLPerceptron<T> &nn) {
    const size_t size = nn.getWeights().size();
    os << "-------input-------\n";
    for (size_t i = 0; i < size; i++) {
      os << "-----weight[" << i << "]-----\n";
      os << nn.getWeights()[i];
      os << "------bias[" << i << "]------\n";
      os << nn.getBiases()[i];
      if (i != size - 1) { os << "-----hidden[" << i << "]-----\n"; }
    }
    os << "-------output------\n";
    return os;
  }

  std::unique_ptr<MLPBase> makeNeuralNetwork(FloatingPrecision precision);

}   // namespace nnet