#pragma once
#include <iostream>
#include <vector>

namespace nn {

// We could also use templated classes
using real = float;

enum class ActivationFunctionType {
  SIGMOID,
  // TODO: Expand me !
};

class NeuralNetwork {
public:
  // default ctor, should create a network with a single layer
  // and random weights/biases
  NeuralNetwork();

  // Not sure we should be able to copy networks ?
  NeuralNetwork(const NeuralNetwork &) = delete;
  NeuralNetwork &operator=(const NeuralNetwork &) = delete;

  // Should throw if the number of input is incorrect
  template <typename iterator>
  std::vector<real> runOnInput(iterator begin, iterator end) const;

  // Should reuse loadFromStream
  void loadFromFile(const std::string &fileName);

  /* Saves the network to a file in binary format, including :
  / - the number of layers
  / - the number of neurons per layer
  / - the weights of each neuron
  / - the biases of each neuron
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

  void randomizeWeights();
  void randomizeBiases();

private:
  // TODO: Add members
};

} // namespace nn