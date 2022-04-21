#include "Utils.hpp"
#include <iostream>
#include <malloc.h>

namespace utils {

  void error(const std::string &msg) noexcept { error(msg.c_str()); }

  void error(const char *msg) noexcept {
    std::cerr << msg << std::endl;
    exit(1);
  }

}   // namespace utils