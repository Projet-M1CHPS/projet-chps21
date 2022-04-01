#include "ActivationFunction.hpp"

namespace af {

  ActivationFunctionType strToAFType(const std::string &str) {
    static const std::unordered_map<std::string, ActivationFunctionType> map{
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
    static const std::unordered_map<ActivationFunctionType, std::string> map{
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

  std::pair<cl::Kernel, cl::Kernel> getAFKernelFromType(ActivationFunctionType type,
                                                        utils::clWrapper &wrapper) {
    auto &map = wrapper.getKernels();
    switch (type) {
      case ActivationFunctionType::identity:
        return {map.getKernel("ActivationFunction.cl", "identity"),
                map.getKernel("ActivationFunction.cl", "didentity")};
      case ActivationFunctionType::sigmoid:
        return {map.getKernel("ActivationFunction.cl", "sigmoid"),
                map.getKernel("ActivationFunction.cl", "dsigmoid")};
      case ActivationFunctionType::relu:
        return {map.getKernel("ActivationFunction.cl", "relu"),
                map.getKernel("ActivationFunction.cl", "drelu")};
      case ActivationFunctionType::leakyRelu:
        return {map.getKernel("ActivationFunction.cl", "leakyRelu"),
                map.getKernel("ActivationFunction.cl", "dleakyRelu")};
      case ActivationFunctionType::square:
        return {map.getKernel("ActivationFunction.cl", "square"),
                map.getKernel("ActivationFunction.cl", "dsquare")};
      default:
        throw std::invalid_argument("getAFKernelFromType(): unknown activation function");
    }
  }

}   // namespace af