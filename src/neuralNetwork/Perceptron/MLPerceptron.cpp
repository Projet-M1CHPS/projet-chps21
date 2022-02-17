#include "MLPerceptron.hpp"

namespace nnet {

  MLPTopology MLPTopology::fromString(const std::string &str) {
    std::vector<size_t> layers;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ',')) { layers.push_back(std::stoi(token)); }
    return MLPTopology(layers);
  }

  math::FloatMatrix MLPerceptron::predict(math::FloatMatrix const &input) const {
    const size_t nbInput = input.getRows();

    if (nbInput != weights.front().getCols()) {
      throw std::invalid_argument("Invalid number of input");
    }

    auto current_layer = math::FloatMatrix::matMatProdMatAdd(weights[0], input, biases[0]);
    auto afunc = af::getAFFromType(activation_functions[0]).first;
    std::transform(current_layer.cbegin(), current_layer.cend(), current_layer.begin(), afunc);

    for (size_t i = 1; i < weights.size(); i++) {
      current_layer = math::FloatMatrix::matMatProdMatAdd(weights[i], current_layer, biases[i]);

      // Apply activation function on every element of the matrix
      afunc = af::getAFFromType(activation_functions[i]).first;
      std::transform(current_layer.cbegin(), current_layer.cend(), current_layer.begin(), afunc);
    }
    return current_layer;
  }


  void MLPerceptron::setTopology(MLPTopology const &topology) {
    if (topology.empty()) return;
    if (topology.size() < 2) { throw std::invalid_argument("Requires atleast 2 layers"); }

    weights.clear();
    biases.clear();
    for (size_t i = 0; i < topology.size() - 1; i++) {
      // Create a matrix of size (layers[i + 1] x layers[i])
      // So that each weight matrix can be multiplied by the previous layer
      weights.emplace_back(topology[i + 1], topology[i]);
      biases.emplace_back(topology[i + 1], 1);
      activation_functions.push_back(af::ActivationFunctionType::sigmoid);
    }
    this->topology = topology;
  }

  void MLPerceptron::randomizeWeight() {
    for (auto &layer : weights) {
      double x = std::sqrt(2.0 / (double) layer.getRows());
      math::randomize<float>(layer, -x, x);
    }

    for (auto &layer : biases) {
      double x = std::sqrt(2.0 / (double) layer.getRows());
      math::randomize<float>(layer, -x, x);
    }
  }

  std::ostream &operator<<(std::ostream &os, const MLPerceptron &nn) {
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

}   // namespace nnet