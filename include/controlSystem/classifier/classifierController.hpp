#pragma once
#include "classifierCollection.hpp"
#include "classifierInputSet.hpp"
#include "classifierTracker.hpp"
#include "controlSystem/controller.hpp"

namespace control::classifier {

  /** Typedef for the classifier controller parameters
   * with the collection loader
   */
  using CTParams = TrainingParameters<CTCLoader>;

  /** Classifier Training Controller, used for training a classifier model
   *
   */
  class CTController : public Controller<float> {
  public:
    template<class Params, typename = std::enable_if<std::is_base_of<CTParams, Params>::value>>
    explicit CTController(Params const &params) {
      this->params = std::make_unique<Params>(params);
    }

    explicit CTController(std::shared_ptr<CTParams> params) : params(std::move(params)) {}

    /** Runs the training process as defined by the parameters
     *
     * No exception will be thrown if the training process fails
     * Instead, a ControllerResult containing the error message will be returned
     *
     * @param is_verbose
     * @param os
     * @return
     */
    ControllerResult run(bool is_verbose, std::ostream *os) noexcept override;

  private:
    ControllerResult load(bool is_verbose, std::ostream *os);
    ControllerResult create(bool is_verbose, std::ostream *os);
    ControllerResult checkModel(bool is_verbose, std::ostream *os);
    void loadTrainingSet(bool is_verbose, std::ostream *os);

    ControllerResult train(bool is_verbose, std::ostream *os);
    void trainingLoop(bool is_verbose, std::ostream *os, CTracker &stracker);
    void printPostTrainingStats(std::ostream &os, CTracker &stracker);

    std::shared_ptr<CTParams> params;
    std::shared_ptr<ClassifierTrainingCollection> training_collection;
  };
}   // namespace control::classifier