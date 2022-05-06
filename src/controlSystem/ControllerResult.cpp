#include "ControllerResult.hpp"

namespace control {
  ControllerResult::ControllerResult(int code, const std::string &message)
      : message(message), return_code(code) {}

  ControllerResult::ControllerResult(int code, const std::runtime_error &exception)
      : message(exception.what()), return_code(code) {}

}   // namespace control