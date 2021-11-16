#include <ActivationFunction.hpp>

namespace af {

  ActivationFunctionType strToAFType(const std::string &str) {
    const std::unordered_map<const char *, ActivationFunctionType> map{
            {"identity", ActivationFunctionType::identity},
            {"sigmoid", ActivationFunctionType::sigmoid},
            {"relu", ActivationFunctionType::relu},
            {"leakyRelu", ActivationFunctionType::leakyRelu},
            {"square", ActivationFunctionType::square},
    };

    auto pair = map.find(str.c_str());
    if (pair == map.end()) {
      throw std::invalid_argument("strToAFType(): unknown activation function");
    }
    return pair->second;
  }

  std::string AFTypeToStr(ActivationFunctionType type) {
    const std::unordered_map<ActivationFunctionType, const char *> map{
            {ActivationFunctionType::identity, "identity"},
            {ActivationFunctionType::sigmoid, "sigmoid"},
            {ActivationFunctionType::relu, "relu"},
            {ActivationFunctionType::leakyRelu, "leakyRelu"},
            {ActivationFunctionType::square, "square"},
    };

    auto pair = map.find(type);
    if (pair == map.end()) {
      throw std::invalid_argument("AFTypeToStr(): unknown activation function mapping");
    }
    return pair->second;
  }

}   // namespace af