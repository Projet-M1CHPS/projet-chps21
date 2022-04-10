#include "Controller.hpp"

namespace control {
  ControllerResult::ControllerResult(int code, const std::string &message)
      : message(message), return_code(code) {}

  ControllerResult::ControllerResult(int code, const std::runtime_error &exception)
      : message(exception.what()), return_code(code) {}

  Controller::Controller(std::filesystem::path output_path) : output_path(std::move(output_path)) {}
}   // namespace control