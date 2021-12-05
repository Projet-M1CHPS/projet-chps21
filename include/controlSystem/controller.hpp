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

  class CTController : Controller {
  public:
    template<class Params, typename = std::enable_if<std::is_base_of<CTParams, Params>::value>>
    explicit CTController(Params const &params) {
      this->params = std::make_unique<Params>(params);
    }

    explicit CTController(std::shared_ptr<CTParams> params) : params(std::move(params)) {}

    ControllerResult run(bool is_verbose, std::ostream *os) override;

  private:
    ControllerResult load(bool is_verbose, std::ostream *os);
    ControllerResult create(bool is_verbose, std::ostream *os);
    ControllerResult checkModel(bool is_verbose, std::ostream *os);
    void loadTrainingSet(bool is_verbose, std::ostream *os);

    ControllerResult train(bool is_verbose, std::ostream *os);

    std::shared_ptr<CTParams> params;
    std::shared_ptr<nnet::NeuralNetwork<float>> network;
    std::shared_ptr<ClassifierTrainingCollection> training_collection;
  };

}   // namespace control
