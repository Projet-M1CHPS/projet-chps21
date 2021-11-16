#include "NeuralNetwork.hpp"

namespace nnet {

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

}   // namespace nnet