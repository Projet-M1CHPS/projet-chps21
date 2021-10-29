#pragma once
#include "Utils.hpp"
#include <iostream>
#include <cmath>
#include <functional>
#include <utility>

namespace af
{

  /**
 * @brief Enum of supported activation functions
 *
 * Function that are not referenced in this enum cannot be used to prevent
 * errors when serializing a network
 *
 */
  enum class ActivationFunctionType
  {
    identity,
    sigmoid,
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
  ActivationFunctionType strToActivationFunctionType(const std::string &str)
  {
    utils::error("Activation function not supported");
  }

  /**
 * @brief Convert an ActivationFunctionType to a string
 *
 * @param type
 * @return std::string
 */
  std::string activationFunctionTypeToStr(ActivationFunctionType type)
  {
    utils::error("Activation function not supported");
  }


   /**
 * @brief Identity math function
 *
 * @to_do: Move me to a separate file
 *
 * @tparam real
 * @param x
 * @return real
 */
  template <typename real>
  real identity(real x)
  {
    static_assert(std::is_floating_point_v<real>,
                  "Invalid type for sigmoid, expected a floating point type");

    return x;
  }

  /**
 * @brief Delta identity math function
 *
 * @to_do: Move me to a separate file
 *
 * @tparam real
 * @param x
 * @return real
 */
  template <typename real>
  real didentity(real x)
  {
    static_assert(std::is_floating_point_v<real>,
                  "Invalid type for sigmoid, expected a floating point type");

    return 1;
  }


  /**
 * @brief Sigmoid math function
 *
 * @to_do: Move me to a separate file
 *
 * @tparam real
 * @param x
 * @return real
 */
  template <typename real>
  real sigmoid(real x)
  {
    static_assert(std::is_floating_point_v<real>,
                  "Invalid type for sigmoid, expected a floating point type");

    return 1.0 / (1.0 + std::exp(-x));
  }

  /**
 * @brief Delta sigmoid math function
 *
 * @to_do: Move me to a separate file
 *
 * @tparam real
 * @param x
 * @return real
 */
  template <typename real>
  real dsigmoid(real x)
  {
    static_assert(std::is_floating_point_v<real>,
                  "Invalid type for sigmoid, expected a floating point type");

    return sigmoid(x) * (1 - sigmoid(x));
  }


  /**
 * @brief Squarre math function
 *
 * @to_do: Move me to a separate file
 *
 * @tparam real
 * @param x
 * @return real
 */
  template <typename real>
  real square(real x)
  {
    static_assert(std::is_floating_point_v<real>,
                  "Invalid type for square, expected a floating point type");
    return x * x;
  }

  /**
 * @brief Delta squarre math function
 *
 * @to_do: Move me to a separate file
 *
 * @tparam real
 * @param x
 * @return real
 */
  template <typename real>
  real dsquare(real x)
  {
    static_assert(std::is_floating_point_v<real>,
                  "Invalid type for square, expected a floating point type");

    return 2 * x;
  }

  /**
 * @brief Return the function associated with an ActivationFunctionType
 *
 * @tparam real
 * @param type
 * @return std::function<real(real)>
 */
  template <typename real>
  std::pair<std::function<real(real)>, std::function<real(real)>>
  getAFFromType(ActivationFunctionType type)
  {
    static_assert(
        std::is_floating_point_v<real>,
        "Invalid type for activation function, expected a floating point type");

    switch (type)
    {
    case ActivationFunctionType::identity:
      return std::make_pair(identity<real>, didentity<real>);
    case ActivationFunctionType::sigmoid:
      return std::make_pair(sigmoid<real>, dsigmoid<real>);
    case ActivationFunctionType::square:
      return std::make_pair(square<real>, dsquare<real>);
    default:
      utils::error("Activation function not supported");
    }
  }

} // namespace af