#pragma once

#include "Utils.hpp"
#include <iostream>
#include <cmath>
#include <functional>
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

/**
 * @brief Convert a string to an ActivationFunctionType
 *
 * @param str
 * @return ActivationFunctionType
 */
  ActivationFunctionType strToAFType(const std::string &str);

/**
 * @brief Convert an ActivationFunctionType to a string
 *
 * @param type
 * @return std::string
 */
  std::string AFTypeToStr(ActivationFunctionType type);

  template<typename real>
  real identity(real x) {
    static_assert(std::is_floating_point_v<real>,
                  "identity(): Invalid type, expected a floating point type");

    return x;
  }

  template<typename real>
  real didentity(real x) {
    static_assert(std::is_floating_point_v<real>,
                  "didentity(): Invalid type, expected a floating point type");

    return 1;
  }

  template<typename real>
  real sigmoid(real x) {
    static_assert(std::is_floating_point_v<real>,
                  "sigmoid(): Invalid type, expected a floating point type");

    return 1.0 / (1.0 + std::exp(-x));
  }

  template<typename real>
  real dsigmoid(real x) {
    static_assert(std::is_floating_point_v<real>,
                  "dsigmoid(): Invalid type, expected a floating point type");

    return sigmoid(x) * (1 - sigmoid(x));
  }

  template<typename real>
  real relu(real x) {
    static_assert(std::is_floating_point_v<real>,
                  "relu(): Invalid type, expected a floating point type");

    return (x <= 0) ? 0.0 : x;
  }

  template<typename real>
  real drelu(real x) {
    static_assert(std::is_floating_point_v<real>,
                  "drelu(): Invalid type, expected a floating point type");

    if (x == 0.0) {
      throw std::invalid_argument("Relu undefined on x = 0.0");
    }

    return (x < 0) ? 0.0 : 1;
  }

  template<typename real>
  real leakyRelu(real x) {
    static_assert(std::is_floating_point_v<real>,
                  "leakyRelu(): Invalid type, expected a floating point type");

    return (x < 0) ? (0.01 * x) : x;
  }

  template<typename real>
  real dleakyRelu(real x) {
    static_assert(std::is_floating_point_v<real>,
                  "dleakyRelu(): Invalid type, expected a floating point type");

    return (x < 0) ? 0.01 : 1;
  }

  template<typename real>
  real square(real x) {
    static_assert(std::is_floating_point_v<real>,
                  "square(): Invalid type, expected a floating point type");
    return x * x;
  }

  template<typename real>
  real dsquare(real x) {
    static_assert(std::is_floating_point_v<real>,
                  "dsquare(): Invalid type, expected a floating point type");

    return 2 * x;
  }

  /**
 * @brief Return the function associated with an ActivationFunctionType
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
            "getAFFromType(): Invalid type, expected a floating point type");

    const std::unordered_map<ActivationFunctionType, std::pair<std::function<real(real)>, std::function<real(
            real)>>> map{
            {ActivationFunctionType::identity,  {identity<real>,  didentity<real>}},
            {ActivationFunctionType::sigmoid,   {sigmoid<real>,   dsigmoid<real>}},
            {ActivationFunctionType::relu,      {relu<real>,      drelu<real>}},
            {ActivationFunctionType::leakyRelu, {leakyRelu<real>, dleakyRelu<real>}},
            {ActivationFunctionType::square,    {square<real>,    dsquare<real>}},
    };
    auto pair = map.find(type);
    if (pair == map.end()) {
      throw std::invalid_argument("getAFFromType(): Unknown activation functions");
    }
    return pair->second;
  }

} // namespace af