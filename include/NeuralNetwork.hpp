#pragma once
#include "Matrix.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <type_traits>
#include <vector>

namespace nnet {

enum class ActivationFunctionType {
  sigmoid,
  // TODO: Expand me !
};

template <typename real> real sigmoid(real x) {
  static_assert(std::is_floating_point_v<real>,
                "Invalid type for sigmoid, expected a floating point type");

  return 1.0 / (1.0 + std::exp(-x));
}

// TODO: Implement additional functions
// (Add them to the enum + getAFFromType())

template <typename real>
std::function<real(real)> getAFFromType(ActivationFunctionType type) {
  static_assert(
      std::is_floating_point_v<real>,
      "Invalid type for activation function, expected a floating point type");

  switch (type) {
  case ActivationFunctionType::sigmoid:
    return sigmoid<real>;
  }
}

template <typename real> class NeuralNetwork {
public:
  // default ctor, should create a network with a single layer
  NeuralNetwork();

  // Not sure we should be able to copy networks ?
  NeuralNetwork(const NeuralNetwork &) = delete;
  NeuralNetwork &operator=(const NeuralNetwork &) = delete;

  // Should throw if the number of input is incorrect
  template <typename iterator>
  std::vector<real> forward(iterator begin, iterator end) const;

  void backward();

  // Should reuse loadFromStream
  void loadFromFile(const std::string &fileName);

  /* Saves the network to a file in binary format, including :
  / - the number of layers
  / - the number of neurons per layer
  / - the weights of each neuron
  / - the activation function
  / Should throw an exception on error
  / TODO: Define a good format for a network
  */
  void loadFromStream(std::istream &stream);

  // Should reuse saveToStream
  void saveToFile(const std::string &fileName) const;
  // Should throw an exception on error
  void saveToStream(std::ostream &stream) const;

  void setLayersSize(std::vector<unsigned int> layers);
  std::vector<unsigned int> getLayersSize() const;

  // Override the default activation function
  void setActivationFunction(ActivationFunctionType type);
  ActivationFunctionType getActivationFunction() const;

  void randomizeSynapses(int seed);

private:
  // TODO: Add members

  std::vector<math::Matrix<real>> matrices;
};

} // namespace nnet