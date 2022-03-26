#include "Controller.hpp"

namespace control {
  ControllerResult::ControllerResult(int code, const std::string &message)
      : return_code(code), message(message) {}

  ControllerResult::ControllerResult(int code, const std::runtime_error &exception)
      : return_code(code), message(exception.what()) {}

  Controller::Controller(std::filesystem::path ouput_path) : output_path(output_path) {}
}   // namespace control