#pragma once

#include "Utils.hpp"
#include "clUtils/clWrapper.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <utility>

namespace af {

  /**
   * @brief Enum of supported activation functions
   *
   * Function that are not referenced in this enum cannot be used to prevent
   * errors when serializing a network
   *
   */
  enum class ActivationFunctionType {
    identity,
    sigmoid,
    relu,
    leakyRelu,
    // TODO: Expand me !

    // Debug
    square
  };


  ActivationFunctionType strToAFType(const std::string &str);
  std::string AFTypeToStr(ActivationFunctionType type);

  /* We assume that every activation function should operate on FP values
   * Henceforth we add static assert at the beginning of every AF
   *
   * Furthermore, every activation function should be linked to its derivative counterpart
   */
  inline float identity(float x) { return x; }

  inline float didentity(float x) { return 1; }

  inline float sigmoid(float x) { return 1.0 / (1.0 + std::exp(-x)); }

  inline float dsigmoid(float x) { return sigmoid(x) * (1 - sigmoid(x)); }

  inline float relu(float x) { return (x <= 0) ? 0.0 : x; }

  inline float drelu(float x) {
    if (x == 0.0) { throw std::invalid_argument("Relu undefined on x = 0.0"); }

    return (x < 0) ? 0.0 : 1;
  }

  inline float leakyRelu(float x) { return (x < 0) ? (0.01 * x) : x; }

  inline float dleakyRelu(float x) { return (x < 0) ? 0.01 : 1; }

  inline float square(float x) { return x * x; }

  inline float dsquare(float x) { return 2 * x; }

  /**
   * @brief Return the function pair associated with an ActivationFunctionType
   * in the form (func, dfunc)
   *
   * @tparam float
   * @param type
   * @return std::function<float(float)>
   */
  inline std::pair<std::function<float(float)>, std::function<float(float)>>
  getAFFromType(ActivationFunctionType type) {
    const std::unordered_map<ActivationFunctionType,
                             std::pair<std::function<float(float)>, std::function<float(float)>>>
            map{
                    {ActivationFunctionType::identity, {identity, didentity}},
                    {ActivationFunctionType::sigmoid, {sigmoid, dsigmoid}},
                    {ActivationFunctionType::relu, {relu, drelu}},
                    {ActivationFunctionType::leakyRelu, {leakyRelu, dleakyRelu}},
                    {ActivationFunctionType::square, {square, dsquare}},
            };
    auto pair = map.find(type);
    if (pair == map.end()) {
      throw std::invalid_argument("getAFFromType(): Unknown activation function");
    }
    return pair->second;
  }

  std::pair<cl::Kernel, cl::Kernel> getAFKernelFromType(ActivationFunctionType type,
                                                        utils::clWrapper &wrapper);

}   // namespace af