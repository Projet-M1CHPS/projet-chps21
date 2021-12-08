#pragma once

#include "ActivationFunction.hpp"
#include "Matrix.hpp"
#include "TrainingMethod.hpp"
#include "Utils.hpp"
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


  FloatingPrecision strToFPrecision(const std::string &str);
  const char *fPrecisionToStr(FloatingPrecision fp);

  /**
   * @brief Convert a static type to the corresponding enum type
   *
   * @tparam typename
   * @return FloatingPrecision
   */
  template<typename T>
  FloatingPrecision getFPPrecision() {
    if constexpr (std::is_same_v<float, T>) {
      return FloatingPrecision::float32;
    } else if constexpr (std::is_same_v<double, T>) {
      return FloatingPrecision::float64;
    } else {
      // dirty trick to prevent the compiler for evaluating the else branch
      static_assert(!sizeof(T), "Invalid floating point type");
    }
  }

  /**
   * @brief Interface for a neural network
   *
   */
  class NeuralNetworkBase {
  public:
    NeuralNetworkBase() = delete;

    virtual ~NeuralNetworkBase() = default;

    [[nodiscard]] FloatingPrecision getPrecision() const { return precision; }

    virtual void setLayersSize(std::vector<size_t> const &layers) = 0;

    [[nodiscard]] virtual size_t getOutputSize() const = 0;

    [[nodiscard]] virtual size_t getInputSize() const = 0;

    [[nodiscard]] virtual std::vector<size_t> getTopology() const = 0;

    virtual void setActivationFunction(af::ActivationFunctionType type) = 0;

    virtual void setActivationFunction(af::ActivationFunctionType af, size_t layer) = 0;

    [[nodiscard]] virtual const std::vector<af::ActivationFunctionType> &
    getActivationFunctions() const = 0;

    virtual void randomizeSynapses() = 0;

  protected:
    // The precision has no reason to change, so no need for setter method
    // And the ctor can be protected
    explicit NeuralNetworkBase(FloatingPrecision precision) : precision(precision) {}

    FloatingPrecision precision = FloatingPrecision::float32;
  };

  /**
   * @brief A neural network that supports most fp precision as template
   * parameters
   *
   * @tparam real
   */
  template<typename real>
  class NeuralNetwork final : public NeuralNetworkBase {
  public:
    using value_type = real;

    /**
     * @brief Construct a new Neural Network object with no layer
     *
     */
    NeuralNetwork() : NeuralNetworkBase(getFPPrecision<real>()) {}

    NeuralNetwork(const NeuralNetwork &other) : NeuralNetworkBase(getFPPrecision<real>()) {
      *this = other;
    }

    NeuralNetwork(NeuralNetwork &&other) noexcept : NeuralNetworkBase(getFPPrecision<real>()) {
      *this = std::move(other);
    }

    NeuralNetwork &operator=(const NeuralNetwork &) = default;

    NeuralNetwork &operator=(NeuralNetwork &&other) noexcept {
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
    void setLayersSize(std::vector<size_t> const &layers) override {
      if (layers.empty()) return;
      if (layers.size() < 2) { throw std::runtime_error("Requires atleast 2 layers"); }

      weights.clear();
      biases.clear();
      for (size_t i = 0; i < layers.size() - 1; i++) {
        // Create a matrix of size (layers[i + 1] x layers[i])
        // So that each weight matrix can be multiplied by the previous layer
        weights.push_back(math::Matrix<real>(layers[i + 1], layers[i]));
        biases.push_back(math::Matrix<real>(layers[i + 1], 1));
        activation_functions.push_back(af::ActivationFunctionType::sigmoid);
      }
    }

    [[nodiscard]] size_t getOutputSize() const override {
      // The last operation is <Mm x Mn> * <Om * On>
      // So the output is <Mm * On> where On is 1

      if (weights.empty()) { return 0; }

      return weights.back().getRows();
    }

    [[nodiscard]] size_t getInputSize() const override {
      // The first operation is <Mm x Mn> * <Om * On>
      // So the input is of size <Mn * On> where On is 1
      if (weights.empty()) { return 0; }

      return weights.front().getCols();
    }

    [[nodiscard]] std::vector<size_t> getTopology() const override {
      std::vector<size_t> res;

      res.reserve(weights.size() + 1);

      res.push_back(getInputSize());
      for (auto &w : weights) { res.push_back(w.getRows()); }
      return res;
    }

    void setActivationFunction(af::ActivationFunctionType type) override {
      for (auto &activation_function : activation_functions) { activation_function = type; }
    }

    void setActivationFunction(af::ActivationFunctionType af, size_t layer) override {
      if (layer >= weights.size()) { throw std::invalid_argument("Invalid layer"); }

      activation_functions[layer] = af;
    }

    [[nodiscard]] const std::vector<af::ActivationFunctionType> &
    getActivationFunctions() const override {
      return activation_functions;
    }

    /**
     * @brief Randomizes the weights and biases of the network
     *
     * @param seed
     */
    void randomizeSynapses() override {
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
  std::ostream &operator<<(std::ostream &os, const NeuralNetwork<T> &nn) {
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

  std::unique_ptr<NeuralNetworkBase> makeNeuralNetwork(FloatingPrecision precision);

}   // namespace nnet