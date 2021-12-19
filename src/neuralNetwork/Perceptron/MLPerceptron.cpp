#include "MLPerceptron.hpp"

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

  std::unique_ptr<MLPBase> makeNeuralNetwork(FloatingPrecision precision) {
    switch (precision) {
      case FloatingPrecision::float32:
        return std::make_unique<MLPerceptron<float>>();
      case FloatingPrecision::float64:
        return std::make_unique<MLPerceptron<double>>();
      default:
        return nullptr;
    }
  }

}   // namespace nnet