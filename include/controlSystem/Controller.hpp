#pragma once
#include <filesystem>
#include <iostream>

namespace control {

  /**
   * @brief Stores the result of a controller. Since the controller is not allowed to throw
   * exceptions, it should store errors in this class.
   */
  class ControllerResult {
  public:
    ControllerResult(int code, const std::string &message);
    ControllerResult(int code, const std::runtime_error &exception);

    /**
     * @brief Returns the message associated of the result.
     * @return
     */
    const std::string &getMessage() const { return message; }

    /**
     * @brief Return the code associated with the result.
     * @return
     */
    int getCode() const { return return_code; }

    /**
     * @brief Returns true if the return code is 0.
     * @return
     */
    operator bool() const { return return_code == 0; }

  private:
    std::string message;
    int return_code;
  };

  class Controller {
  public:
    explicit Controller(std::filesystem::path ouput_path);
    virtual ~Controller() = default;

    virtual ControllerResult run() noexcept = 0;

    std::filesystem::path getOutputPath() const { return output_path; }

  private:
    std::filesystem::path output_path;
  };

}   // namespace control
