#include "NeuralNetwork.hpp"
#include <functional>
#include <cmath>

namespace nn {
template <typename real> real sigmoid(real x) {
  return 1.0 / (1.0 + std::exp(-x));
}

// TODO: Implement additional functions
// (Add them to the ActivationFunctionType enum + getAFFromType())

template <typename real>
std::function<real(real)> getAFFromType(ActivationFunctionType type) {
  switch (type) {
  case ActivationFunctionType::SIGMOID:
    return sigmoid<real>;
  }
}

//TODO : implement NeuralNetwork class

} // namespace nn