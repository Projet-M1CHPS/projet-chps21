
#include "controller.hpp"
#include <iostream>

namespace control {

  std::ostream &operator<<(std::ostream &os, const control::ControllerResult &res) {
    res.print(os);
    return os;
  }

}   // namespace control
