#pragma once

#include <controlSystem/controllerParameters.hpp>

namespace control {

  class ControllerResult {
  public:
    ControllerResult(bool res, std::string msg) : result(res), message(std::move(msg)) {}

    virtual ~ControllerResult() = default;

    explicit operator bool() const { return result; }
    [[nodiscard]] std::string const &getMessage() const { return message; }

    virtual void print(std::ostream &os) const {
      os << "Result: " << result << " Message: " << message;
    }

  private:
    bool result;
    std::string message;
  };

  class Controller {
  public:
    virtual ControllerResult run(bool is_verbose, std::ostream *os) = 0;

  protected:
  };
}   // namespace control
