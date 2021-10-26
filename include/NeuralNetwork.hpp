#pragma once
#include "Matrix.hpp"
#include "Utils.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <type_traits>
#include <vector>

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
ActivationFunctionType strToActivationFunctionType(const std::string &str) {
  utils::error("Activation function not supported");
}

/**
 * @brief Convert an ActivationFunctionType to a string
 *
 * @param type
 * @return std::string
 */
std::string activationFunctionTypeToStr(ActivationFunctionType type) {
  utils::error("Activation function not supported");
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
  FloatingPrecision getPrecision() { return precision; }

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
  /**
   * @brief Construct a new Neural Network object with no layer
   *
   */
  NeuralNetwork() : NeuralNetworkBase(getFPPrecision<real>()) {}

  /**
   * @brief Construct a copy of an existing neural network
   *
   * @param other
   */
  NeuralNetwork(const NeuralNetwork &other)
      : NeuralNetworkBase(getFPPrecision<real>()) {
    *this = other;
  }
  NeuralNetwork &operator=(const NeuralNetwork &) = default;

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

  // TODO: implement the backpropagation algorithm
  void backward();

  template <typename iterator>
  void train(iterator begin_input, iterator end_input, iterator begin_target,
             iterator end_target) const {
    std::vector<math::Matrix<real>> layers;

    // Forward
    // ------------------------------------------------------------------------
    const size_t nbInput = std::distance(begin_input, end_input);

    if (nbInput != weights.front().getCols()) {
      throw std::invalid_argument("Invalid number of input");
    }

    math::Matrix<real> current_layer(nbInput, 1);
    std::copy(begin_input, end_input, current_layer.begin());

    layers.push_back(current_layer);

    for (size_t i = 0; i < weights.size(); i++) {

      current_layer = forwardOnce(current_layer, i);

      layers.push_back(current_layer);
    }

    // Backward
    // ------------------------------------------------------------------------

    const size_t nbTarget = std::distance(begin_target, end_target);
    if (nbTarget != getOutputSize()) {
      throw std::invalid_argument("Invalid number of target");
    }

    math::Matrix<real> target(nbTarget, 1);
    std::copy(begin_target, end_target, target.begin());

    std::vector<math::Matrix<real>> errors;
    math::Matrix<real> current_error = target - current_layer;
    errors.push_back(current_error);

    for (long i = weights.size() - 1; i >= 0; i--) {
      math::Matrix<real> wt = weights[i].transpose();
      current_error = wt * current_error;
      errors.push_back(current_error);
    }

    std::cout << "size " << errors.size() << std::endl;
    for (auto &m : errors) {
      for (auto &j : m)
        std::cout << j << std::endl;
      std::cout << std::endl;
    }
  }

  /**
   * @brief Take a vector of sizes correspondig to the number of neurons
   * in each layer and build the network accordingly. Note that weights are not
   * initialized after this.
   *
   * @param layers
   */
  void setLayersSize(std::vector<size_t> layers) {

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

  /**
   * @brief Returns the size of the output layer
   *
   * @return size_t
   */
  size_t getOutputSize() const {
    // The last operation is <Mm x Mn> * <Om * On>
    // So the output is <Mm * On> where On is 1

    if (weights.empty()) {
      return 0;
    }

    return weights.back().getRows();
  }

  /**
   * @brief Returns the size of the input layer
   *
   * @return size_t
   */
  size_t getInputSize() const {
    // The first operation is <Mm x Mn> * <Om * On>
    // So the input is of size <Mn * On> where On is 1
    if (weights.empty()) {
      return 0;
    }

    return weights.front().getCols();
  }

  /**
   * @brief Returns a vector containing the size of each layer
   *
   * @return std::vector<size_t>
   */
  std::vector<size_t> getLayersSize() const {
    std::vector<size_t> res(weights.size());

    for (size_t i = 0; i < weights.size(); i++) {
      res[i] = weights[i].getCols();
    }
    return res;
  }

  /**
   * @brief Set the Activation Function of a given layer
   *
   * @param type
   * @param layer
   */
  void setActivationFunction(ActivationFunctionType type) {

    for (size_t i = 0; i < activation_functions.size(); i++) {
      activation_functions[i] = type;
    }
  }

  /**
   * @brief Set the Activation Function of a given layer
   *
   * @param type
   * @param layer
   */
  void setActivationFunction(ActivationFunctionType af, size_t layer) {
    if (layer >= weights.size()) {
      throw std::invalid_argument("Invalid layer");
    }

    activation_functions[layer] = af;
  }

  /**
   * @brief Randomizes the weights and biases of the network
   *
   * @param seed
   */
  void randomizeSynapses() {
    for (auto &layer : weights) {
      utils::random::randomize<real>(layer, 0, 1);
    }

    for (auto &layer : weights) {
      utils::random::randomize<real>(layer, 0, 1);
    }
  }

  /**
   * @brief Geter for the weights
   */
  std::vector<math::Matrix<real>> &getWeights() { return weights; }

  /**
   * @brief Geter for the biases matrices
   *
   */
  std::vector<math::Matrix<real>> &getBiaises() { return biases; }

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

/**
 * @brief Main class for serializing and deserializing neural networks
 * Also offers utility for outputing data about the network in a json format
 *
 */
class NeuralNetworkSerializer {
public:
  NeuralNetworkSerializer(NeuralNetworkBase &network) : network(&network) {}

  void saveToFile();
  void saveToStream();

  void loadFromFile();
  void loadFromStream();

private:
  NeuralNetworkBase *network;
};

} // namespace nnet