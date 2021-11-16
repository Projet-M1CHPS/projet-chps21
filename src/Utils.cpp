#include "Utils.hpp"
#include <iostream>

namespace utils {

  void error(const std::string &msg) noexcept { error(msg.c_str()); }

  void error(const char *msg) noexcept {
    std::cerr << msg << std::endl;
    exit(1);
  }

  // Delegate to the other constructor
  IOException::IOException(const std::string &msg) noexcept
      : IOException(msg.c_str()) {}

  // Prefix the message with "IOException: " and forwardd it to the base
  // constructor
  IOException::IOException(const char *msg) noexcept
      : std::runtime_error(std::string("IOException: ") + msg) {}

}   // namespace utils