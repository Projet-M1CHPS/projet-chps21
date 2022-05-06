#pragma once
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <type_traits>

namespace utils {

  /**
   * @brief Exit the program with an error message. Should only be used where a
   * throw is not enough
   *
   * @param msg
   */
  [[noreturn]] void error(const std::string &msg) noexcept;

  /**
   * @brief Exit the program with an error message. Should only be used where a
   * throw is not enough
   *
   * @param msg
   */
  [[noreturn]] void error(const char *msg) noexcept;

}   // namespace utils