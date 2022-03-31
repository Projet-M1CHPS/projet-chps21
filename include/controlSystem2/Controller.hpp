#pragma once
#include "clUtils/clWrapper.hpp"
#include <filesystem>
#include <iostream>

namespace control {

  class ControllerResult {
  public:
    ControllerResult(int code, const std::string &message);
    ControllerResult(int code, const std::runtime_error &exception);

    const std::string &getMessage() const { return message; }
    int getCode() const { return return_code; }

  private:
    std::string message;
    int return_code;
  };

  class Controller {
  public:
    explicit Controller(std::filesystem::path ouput_path);

    virtual ~Controller() = default;
    virtual ControllerResult run() = 0;

    std::filesystem::path getOutputPath() const { return output_path; }

  private:
    std::filesystem::path output_path;
  };

}   // namespace control
