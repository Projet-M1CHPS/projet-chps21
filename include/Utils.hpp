#pragma once
#include "Matrix.hpp"
#include <iostream>
#include <random>
#include <type_traits>

namespace utils {

void error(const std::string &msg) noexcept;

void error(const char *msg) noexcept;

namespace random {

template <typename T> void randomize(math::Matrix<T> &matrix, T min, T max) {

  std::random_device rd;
  std::mt19937 gen(rd());

  static constexpr bool handled =
      std::is_floating_point_v<T> or std::is_integral_v<T>;
  static_assert(handled, "Type not supported");

  if constexpr (std::is_floating_point_v<T>) {
    std::uniform_real_distribution<> dis(min, max);
    for (auto& elem : matrix)
      elem = dis(gen);

  } else if constexpr (std::is_integral_v<T>) {
    std::uniform_int_distribution<> dis(min, max);
    for (auto& elem : matrix)
      elem = dis(gen);
  }
}

} // namespace random

// Generic IO exception
class IOException : public std::runtime_error {
public:
  IOException(const std::string &msg) noexcept;
  IOException(const char *msg) noexcept;
};

} // namespace utils