#pragma once

#include "tscl.hpp"
#include <controlSystem/controllerParameters.hpp>

namespace control {

  /** @brief Utility class used for returning a RunController output
   *
   *  Since no exception should reach the python layer, this class is used
   *  for returning messages and storing any caught errors
   */
  class ControllerResult {
    friend std::ostream &operator<<(std::ostream &, const ControllerResult &);

  public:
    ControllerResult(size_t status, std::string msg) : status(status), message(std::move(msg)) {}
    virtual ~ControllerResult() = default;

    /** Returns true if the status is != 0
     *
     * @return
     */
    explicit operator bool() const { return status; }

    /* Returns the message associated with the result
     * Can be an error or a success report
     */
    [[nodiscard]] std::string const &getMessage() const { return message; }

    /** Return the status code associated with the result
     *
     * @return
     */
    [[nodiscard]] size_t getStatus() const { return status; }

  private:
    /** Prints the content of the result as "<Status>: <Message>"
     *
     * @param os
     */
    virtual void print(std::ostream &os) const {
      os << "Status: " << status << " Message: " << message;
    }

    size_t status;
    std::string message;
  };

  /** @brief Interface for controllers
   *
   */
  class RunController {
  public:
    explicit RunController(nnet::Model<float> &model) : model(&model) {}

    /** Controllers should not be copied
     * This may change in the future if a case where this is needed is found
     *
     * @param other
     */
    RunController(RunController const &other) = delete;
    RunController(RunController &&other) = delete;

    virtual ~RunController() = default;

    /** Starts a run using the stored model, an register a callback in the exit handler
     * in case of an unexpected exit.
     *
     * If the callback is called, the controller will attempt to dump the @Model to disk.
     * On error, a @ControllerResult is returned.
     * This is done to prevent exceptions from reaching the python layer.
     * This also means that every exception is caught, handled or not
     * FIXME: implement me
     *
     * @param e_handler Exception handler to be called on unexpected exit
     * @return
     */
    // virtual ControllerResult run(tscl::ExitHandler &e_handler) noexcept = 0;

    /** Starts a run using the stored @Model.
     *
     * On error, a @ControllerResult is returned.
     * This is done to prevent exceptions from reaching the python layer.
     * This also means that every exception is caught, handled or not
     *
     * @return
     */
    virtual ControllerResult run() noexcept = 0;

    /** Return the controller's @Model
     *
     * Returns nullptr if no network is set
     *
     * @return
     */
    [[nodiscard]] nnet::Model<float> *getModel() { return model; }
    [[nodiscard]] nnet::Model<float> const *getModel() const { return model; }

  protected:
    nnet::Model<float> *model;
  };
}   // namespace control
