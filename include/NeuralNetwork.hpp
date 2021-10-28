#pragma once
#include "Matrix.hpp"
#include "Utils.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <type_traits>
#include <vector>

#define alpha .1

namespace nnet {

/**
 * @brief Enum of supported activation functions
 *
 * Function that are not referenced in this enum cannot be used to prevent
 * errors when serializing a network
 *
 */
enum class ActivationFunctionType {
  sigmoid,
  // TODO: Expand me !

  // Debug
  square
};

/**
 * @brief Convert a string to an ActivationFunctionType
 *
 * @param str
 * @return ActivationFunctionType
 */
ActivationFunctionType strToAFType(const std::string &str) {

  if (str == "sigmoid") {
    return ActivationFunctionType::sigmoid;
  } else if (str == "square") {
    return ActivationFunctionType::square;
  } else {
    throw std::runtime_error("Unknown activation function type");
  }
}

/**
 * @brief Convert an ActivationFunctionType to a string
 *
 * @param type
 * @return std::string
 */
std::string AFTypeToStr(ActivationFunctionType type) {
  switch (type) {
  case ActivationFunctionType::sigmoid:
    return "sigmoid";
  case ActivationFunctionType::square:
    return "square";
  default:
    throw std::runtime_error("Unknown activation function type");
  }
}

/**
 * @brief Sigmoid math function
 *
 * @to_do: Move me to a separate file
 *
 * @tparam real
 * @param x
 * @return real
 */
template <typename real> real sigmoid(real x) {
  static_assert(std::is_floating_point_v<real>,
                "Invalid type for sigmoid, expected a floating point type");

  return 1.0 / (1.0 + std::exp(-x));
}

/**
 * @brief Delta sigmoid math function
 *
 * @to_do: Move me to a separate file
 *
 * @tparam real
 * @param x
 * @return real
 */
template <typename real> real dsigmoid(real x) {
  static_assert(std::is_floating_point_v<real>,
                "Invalid type for sigmoid, expected a floating point type");

  return sigmoid(x) * (1 - sigmoid(x));
}

/**
 * @brief Squarre math function
 *
 * @to_do: Move me to a separate file
 *
 * @tparam real
 * @param x
 * @return real
 */
template <typename real> real square(real x) {
  static_assert(std::is_floating_point_v<real>,
                "Invalid type for square, expected a floating point type");

  return x * x;
}

/**
 * @brief Delta squarre math function
 *
 * @to_do: Move me to a separate file
 *
 * @tparam real
 * @param x
 * @return real
 */
template <typename real> real dsquare(real x) {
  static_assert(std::is_floating_point_v<real>,
                "Invalid type for square, expected a floating point type");

  return 2 * x;
}

/**
 * @brief Return the function associated with an ActivationFunctionType
 *
 * @tparam real
 * @param type
 * @return std::function<real(real)>
 */
template <typename real>
std::function<real(real)> getAFFromType(ActivationFunctionType type) {
  static_assert(
      std::is_floating_point_v<real>,
      "Invalid type for activation function, expected a floating point type");

  switch (type) {
  case ActivationFunctionType::sigmoid:
    return sigmoid<real>;
  case ActivationFunctionType::square:
    return square<real>;
  default:
    utils::error("Activation function not supported");
  }
}

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

/**
 * @brief
 *
 * @param str
 * @return FloatingPrecision
 */
FloatingPrecision strToFPrecision(const std::string &str) {
  if (str == "float32") {
    return FloatingPrecision::float32;
  } else if (str == "float64") {
    return FloatingPrecision::float64;
  } else {
    throw std::invalid_argument("Invalid floating precision");
  }
}

const char *fPrecisionToStr(FloatingPrecision fp) {
  switch (fp) {
  case FloatingPrecision::float32:
    return "float32";
  case FloatingPrecision::float64:
    return "float64";
  default:
    throw std::invalid_argument("Invalid floating precision");
  }
}

/**
 * @brief Convert a static type to the corresponding enum type
 *
 * @tparam ypename
 * @return FloatingPrecision
 */
template <typename T> FloatingPrecision getFPPrecision() {
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
 * @brief Base class for the neural network
 *
 */
class NeuralNetworkBase {
public:
  NeuralNetworkBase() = delete;
  virtual ~NeuralNetworkBase() = default;

  FloatingPrecision getPrecision() const { return precision; }

  virtual void setLayersSize(std::vector<size_t> layers) = 0;
  virtual size_t getOutputSize() const = 0;
  virtual size_t getInputSize() const = 0;
  virtual std::vector<size_t> getLayersSize() const = 0;
  virtual void setActivationFunction(ActivationFunctionType type) = 0;
  virtual void setActivationFunction(ActivationFunctionType af,
                                     size_t layer) = 0;
  virtual const std::vector<ActivationFunctionType> &
  getActivationFunctions() const = 0;

  virtual void randomizeSynapses() = 0;

protected:
  // The precision has no reason to change, so no need for setter method
  // And the ctor can be protected
  NeuralNetworkBase(FloatingPrecision precision) : precision(precision) {}

  FloatingPrecision precision = FloatingPrecision::float32;
};

/**
 * @brief A neural network that supports most fp precision as template
 * parameters
 *
 * @tparam real
 */
template <typename real> class NeuralNetwork final : public NeuralNetworkBase {
public:
  using value_type = real;

  /**
   * @brief Construct a new Neural Network object with no layer
   *
   */
  NeuralNetwork() : NeuralNetworkBase(getFPPrecision<real>()) {}

  NeuralNetwork(const NeuralNetwork &other)
      : NeuralNetworkBase(getFPPrecision<real>()) {
    *this = other;
  }

  NeuralNetwork(NeuralNetwork &&other)
      : NeuralNetworkBase(getFPPrecision<real>()) {
    *this = std::move(other);
  }

  NeuralNetwork &operator=(const NeuralNetwork &) = default;
  NeuralNetwork &operator=(NeuralNetwork &&other) {
    weights = std::move(other.weights);
    biases = std::move(other.biases);
    activation_functions = std::move(other.activation_functions);

    return *this;
  }

  /**
   * @brief Run the network on a set of input, throwing on error
   *
   *
   * @tparam iterator
   * @param begin
   * @param end
   * @return math::Matrix<real> A vector containing the network's output
   */
  template <typename iterator>
  math::Matrix<real> forward(iterator begin, iterator end) const {

    const size_t nbInput = std::distance(begin, end);

    if (nbInput != weights.front().getCols()) {
      throw std::invalid_argument("Invalid number of input");
    }

    math::Matrix<real> current_layer(nbInput, 1);
    std::copy(begin, end, current_layer.begin());

    for (size_t i = 0; i < weights.size(); i++) {
      current_layer = forwardOnce(current_layer, i);
    }
    return current_layer;
  }

  template <typename iterator>
  void train(iterator begin_input, iterator end_input, iterator begin_target,
             iterator end_target) {
    std::vector<math::Matrix<real>> layers;
    std::vector<math::Matrix<real>> layers_buffer;
    // Forward
    // ------------------------------------------------------------------------
    const size_t nbInput = std::distance(begin_input, end_input);

    if (nbInput != weights.front().getCols()) {
      throw std::invalid_argument("Invalid number of input");
    }

    math::Matrix<real> current_layer(nbInput, 1);
    std::copy(begin_input, end_input, current_layer.begin());

    layers.push_back(current_layer);
    layers_buffer.push_back(current_layer);

    for (size_t i = 0; i < weights.size(); i++) {
      // C = W * C + B
      current_layer = weights[i] * current_layer;
      // Avoid a copy by using the += operator
      current_layer += biases[i];

      layers_buffer.push_back(current_layer);

      // Apply activation function on every element of the matrix
      // C_ij = AF(C_ij)
      std::function<real(real)> af =
          getAFFromType<real>(activation_functions[i]);
      std::for_each(current_layer.cbegin(), current_layer.cend(), af);

      layers.push_back(current_layer);
    }

    // Backward
    // ------------------------------------------------------------------------

    const size_t nbTarget = std::distance(begin_target, end_target);
    if (nbTarget != getOutputSize()) {
      throw std::invalid_argument("Invalid number of target");
    }

    //
    math::Matrix<real> target(nbTarget, 1);
    std::copy(begin_target, end_target, target.begin());

    //
    math::Matrix<real> current_error = target - current_layer;

    for (long i = weights.size() - 1; i >= 0; i--) {
      std::cout << "\ni = " << i << std::endl;

      // calcul de S
      math::Matrix<real> gradient(layers_buffer[i + 1]);
      std::function<real(real)> daf = dsigmoid<real>;
      std::for_each(gradient.cbegin(), gradient.cend(), daf);

      // calcul de S * E
      gradient = gradient * current_error;
      // calcul de (S * E) * alpha
      gradient = gradient * alpha;

      // calcul de ((S * E) * alpha) * Ht
      math::Matrix<real> ht = layers[i].transpose();
      math::Matrix<real> delta_weight = gradient * ht;

      weights[i] = weights[i] + delta_weight;
      biases[i] = biases[i] + gradient;

      math::Matrix<real> wt = weights[i].transpose();
      current_error = wt * current_error;
    }
  }

  /**
   * @brief Take a vector of sizes correspondig to the number of neurons
   * in each layer and build the network accordingly. Note that weightsremains
   * uninitialized after this.
   *
   * @param layers
   */
  virtual void setLayersSize(std::vector<size_t> layers) override {

    if (layers.size() < 2) {
      throw std::invalid_argument("Requires atleast 2 layers");
    }

    weights.clear();
    biases.clear();
    for (size_t i = 0; i < layers.size() - 1; i++) {
      // Create a matrix of size (layers[i + 1] x layers[i])
      // So that each weight matrix can be multiplied by the previous layer
      weights.push_back(math::Matrix<real>(layers[i + 1], layers[i]));
      biases.push_back(math::Matrix<real>(layers[i + 1], 1));
      activation_functions.push_back(ActivationFunctionType::sigmoid);
    }
  }

  virtual size_t getOutputSize() const override {
    // The last operation is <Mm x Mn> * <Om * On>
    // So the output is <Mm * On> where On is 1

    if (weights.empty()) {
      return 0;
    }

    return weights.back().getRows();
  }

  virtual size_t getInputSize() const override {
    // The first operation is <Mm x Mn> * <Om * On>
    // So the input is of size <Mn * On> where On is 1
    if (weights.empty()) {
      return 0;
    }

    return weights.front().getCols();
  }

  virtual std::vector<size_t> getLayersSize() const override {
    std::vector<size_t> res(weights.size());

    for (size_t i = 0; i < weights.size(); i++) {
      res[i] = weights[i].getCols();
    }
    return res;
  }

  virtual void setActivationFunction(ActivationFunctionType type) override {

    for (size_t i = 0; i < activation_functions.size(); i++) {
      activation_functions[i] = type;
    }
  }

  virtual void setActivationFunction(ActivationFunctionType af,
                                     size_t layer) override {
    if (layer >= weights.size()) {
      throw std::invalid_argument("Invalid layer");
    }

    activation_functions[layer] = af;
  }

  virtual const std::vector<ActivationFunctionType> &
  getActivationFunctions() const override {
    return activation_functions;
  }

  /**
   * @brief Randomizes the weights and biases of the network
   *
   */
  virtual void randomizeSynapses() override {
    for (auto &layer : weights) {
      utils::random::randomize<real>(layer, 0, 1);
    }

    for (auto &layer : weights) {
      utils::random::randomize<real>(layer, 0, 1);
    }
  }

  std::vector<math::Matrix<real>> &getWeights() { return weights; }
  const std::vector<math::Matrix<real>> &getWeights() const { return weights; }

  std::vector<math::Matrix<real>> &getBiases() { return biases; }
  const std::vector<math::Matrix<real>> &getBiases() const { return biases; }

private:
  math::Matrix<real> forwardOnce(const math::Matrix<real> &mat,
                                 const size_t index) const {
    // C = W * C + B

    math::Matrix<real> res = weights[index] * mat;
    // Avoid a copy by using the += operator
    res += biases[index];

    // Apply activation function on every element of the matrix
    // C_ij = AF(C_ij)
    std::function<real(real)> af =
        getAFFromType<real>(activation_functions[index]);
    std::for_each(res.cbegin(), res.cend(), af);
    return res;
  }

private:
  std::vector<math::Matrix<real>> weights;
  std::vector<math::Matrix<real>> biases;

  // We want every layer to have its own activation function
  std::vector<ActivationFunctionType> activation_functions;
};
} // namespace nnet