#include "Controller.hpp"

namespace control {
  ControllerResult::ControllerResult(int code, const std::string &message)
      : return_code(code), message(message) {}

  ControllerResult::ControllerResult(int code, const std::runtime_error &exception)
      : return_code(code), message(exception.what()) {}

  Controller::Controller(std::filesystem::path output_path) : output_path(std::move(output_path)) {}
}   // namespace control