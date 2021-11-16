#pragma once

#include "Utils.hpp"
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

  template<typename real>
  real identity(real x) {
    static_assert(std::is_floating_point_v<real>,
                  "Invalid type, expected a floating point type");

    return x;
  }

  template<typename real>
  real didentity(real x) {
    static_assert(std::is_floating_point_v<real>,
                  "Invalid type, expected a floating point type");

    return 1;
  }

  template<typename real>
  real sigmoid(real x) {
    static_assert(std::is_floating_point_v<real>,
                  "Invalid type, expected a floating point type");

    return 1.0 / (1.0 + std::exp(-x));
  }

  template<typename real>
  real dsigmoid(real x) {
    static_assert(std::is_floating_point_v<real>,
                  "Invalid type, expected a floating point type");

    return sigmoid(x) * (1 - sigmoid(x));
  }

  template<typename real>
  real relu(real x) {
    static_assert(std::is_floating_point_v<real>,
                  "Invalid type, expected a floating point type");

    return (x <= 0) ? 0.0 : x;
  }

  template<typename real>
  real drelu(real x) {
    static_assert(std::is_floating_point_v<real>,
                  "Invalid type, expected a floating point type");

    if (x == 0.0) {
      throw std::invalid_argument("Relu undefined on x = 0.0");
    }

    return (x < 0) ? 0.0 : 1;
  }

  template<typename real>
  real leakyRelu(real x) {
    static_assert(std::is_floating_point_v<real>,
                  "Invalid type, expected a floating point type");

    return (x < 0) ? (0.01 * x) : x;
  }

  template<typename real>
  real dleakyRelu(real x) {
    static_assert(std::is_floating_point_v<real>,
                  "Invalid type, expected a floating point type");

    return (x < 0) ? 0.01 : 1;
  }

  template<typename real>
  real square(real x) {
    static_assert(std::is_floating_point_v<real>,
                  "Invalid type, expected a floating point type");
    return x * x;
  }

  template<typename real>
  real dsquare(real x) {
    static_assert(std::is_floating_point_v<real>,
                  "Invalid type, expected a floating point type");

    return 2 * x;
  }

  /**
   * @brief Return the function pair associated with an ActivationFunctionType
   * in the form (func, dfunc)
   *
   * @tparam real
   * @param type
   * @return std::function<real(real)>
   */
  template<typename real>
  std::pair<std::function<real(real)>, std::function<real(real)>>
  getAFFromType(ActivationFunctionType type) {
    static_assert(
            std::is_floating_point_v<real>,
            "Invalid type, expected a floating point type");

    const std::unordered_map<ActivationFunctionType, std::pair<std::function<real(real)>, std::function<real(
                                                                                                  real)>>>
            map{
                    {ActivationFunctionType::identity, {identity<real>, didentity<real>}},
                    {ActivationFunctionType::sigmoid, {sigmoid<real>, dsigmoid<real>}},
                    {ActivationFunctionType::relu, {relu<real>, drelu<real>}},
                    {ActivationFunctionType::leakyRelu, {leakyRelu<real>, dleakyRelu<real>}},
                    {ActivationFunctionType::square, {square<real>, dsquare<real>}},
            };
    auto pair = map.find(type);
    if (pair == map.end()) {
      throw std::invalid_argument("getAFFromType(): Unknown activation function");
    }
    return pair->second;
  }

}   // namespace af