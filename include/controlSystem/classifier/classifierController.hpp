#pragma once
#include "classifierCollection.hpp"
#include "classifierInputSet.hpp"
#include "controlSystem/controller.hpp"

namespace control::classifier {

  using CTParams = TrainingParameters<CTCLoader>;

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
}   // namespace control::classifier