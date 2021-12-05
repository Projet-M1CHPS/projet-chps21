#pragma once

#include <controlSystem/controllerParameters.hpp>

namespace control {

  /** Utility class used for returning a controller's output
   *
   *  Since no exception should reach the python layer, this class is used
   *  for returning messages and storing any caught errors
   */
  class ControllerResult {
    friend std::ostream &operator<<(std::ostream &, const ControllerResult &);

  public:
    ControllerResult(size_t status, std::string msg) : status(status), message(std::move(msg)) {}
    virtual ~ControllerResult() = default;

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
    /** Placeholder method for printing the content of the result
     *
     * @param os
     */
    virtual void print(std::ostream &os) const {
      os << "Status: " << status << " Message: " << message;
    }

    size_t status;
    std::string message;
  };

  /** Base class for all controllers
   *
   */
  template<typename real, typename = std::enable_if<std::is_floating_point<real>::value>>
  class Controller {
  public:
    virtual ~Controller() = default;
    virtual ControllerResult run(bool is_verbose, std::ostream *os) noexcept = 0;

    /** Return the controller's network
     *
     * May return nullptr if no network is loaded
     *
     * @return
     */
    [[nodiscard]] nnet::NeuralNetwork<real> *getNetwork() { return network.get(); }
    [[nodiscard]] nnet::NeuralNetwork<real> const *getNetwork() const { return network.get(); }

    /** Returns a shared_ptr pointing to the controller's network
     *
     * @return
     */
    [[nodiscard]] std::shared_ptr<nnet::NeuralNetwork<real>> getNetworkPtr() { return network; }
    [[nodiscard]] std::shared_ptr<nnet::NeuralNetwork<real> const> getNetworkPtr() const {
      return network;
    }

  protected:
    std::shared_ptr<nnet::NeuralNetwork<real>> network;
  };
}   // namespace control
