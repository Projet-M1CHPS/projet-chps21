
#include "classifierController.hpp"
#include "Network.hpp"
#include "classifierTracker.hpp"
#include "tscl.hpp"

namespace control::classifier {

  /** This function should not throw on error
   *
   */
  ControllerResult CTController::run() noexcept {
    auto &param = *params;
    RunPolicy policy = param.getRunPolicy();

    tscl::logger("CTController running");
    try {
      if (policy == RunPolicy::load) {
        // Tries to load a saved model, and throws an exception if it fails.
        tscl::logger("Loading model...");
        ControllerResult res = load();
        if (not res) return res;
      } else if (policy == RunPolicy::create) {
        tscl::logger("Creating new model...");
        // Create a new model and erase any saved model
        ControllerResult res = create();
        if (not res) return res;
      } else if (policy == RunPolicy::tryLoad) {
        // Tries to load a saved model that fits the given parameters
        // Throws if the load fails or if the parameters don't match
        tscl::logger("Trying to load model...");
        ControllerResult res = load();
        // FIXME: parameter verification
        return res;
      } else if (policy == RunPolicy::loadOrOverwrite) {
        // Tries to load a saved model that fits the given parameters
        // If it fails, creates a new one with the given parameters and overwrite any previously
        // saved model
        tscl::logger("Trying to load model...");
        ControllerResult res = load();
        if (not res) {
          tscl::logger("Loading failed, creating new model...");
          res = create();
          if (not res) return res;
        }
      }
      train();
    } catch (std::exception &e) {
      tscl::logger("CTController failed: " + std::string(e.what()), tscl::Log::Error);
      return {false, e.what()};
    }
    return {true, "Run successful"};
  }

  ControllerResult CTController::load() {
    nnet::MLPModelSerializer<float> serializer;

    model = serializer.load(params->getInputPath());

    if (not model) {
      tscl::logger("Failed to load model", tscl::Log::Error);
      return {false, "Failed to load model"};
    }

    loadCollection();
    auto topology = params->getTopology();
    auto &network = model->getPerceptron();

    if (topology.getOutputSize() != network.getTopology().getOutputSize()) {
      tscl::logger("The loaded model has a different output size than the number of classes found "
                   "in the collection.",
                   tscl::Log::Warning);
      tscl::logger("Training will stop to prevent breaking the model.", tscl::Log::Warning);
      tscl::logger("To continue anyway, please change the number of classes in the collection or "
                   "the number of classes in the model.",
                   tscl::Log::Warning);
      tscl::logger("Stopping run", tscl::Log::Error);
      return {false, "Model topology mismatch"};
    }
    tscl::logger("Succesfully loaded model", tscl::Log::Information);

    return {false, "not implemented"};
  }

  ControllerResult CTController::create() {
    loadCollection();
    auto topology = params->getTopology();
    topology.push_back(training_collection->classCount());

    model = nnet::MLPModelFactory<float>::randomSigReluAlt(nnet::MLPTopology(topology));

    return {true, "Training set loaded"};
  }

  ControllerResult CTController::checkModel() { return {false, "not implemented"}; }

  void CTController::loadCollection() {
    tscl::logger("Loading collection...");
    try {
      if (not training_collection) {
        training_collection = params->getSetLoader().load(params->getInputPath());
        training_collection->shuffleSets(std::random_device{}());
      }
    } catch (std::exception &e) {
      throw std::runtime_error("Error while loading training set: " + std::string(e.what()));
    }
    std::filesystem::create_directory(params->getWorkingPath() / "cache");
  }


  ControllerResult CTController::train() {
    if (not training_collection or not model) {
      throw std::runtime_error("CTController: train called with missing training set or network");
    }

    auto &classes = training_collection->getClasses();
    size_t nclass = training_collection->classCount();
    CTracker stracker(params->getWorkingPath() / "output", classes.begin(), classes.end());


    trainingLoop(stracker);
    printPostTrainingStats(stracker);
    tscl::logger("Training finished", tscl::Log::Information);

    return {true, "Training finished"};
  }

  void CTController::trainingLoop(CTracker &stracker) {
    double initial_learning_rate = 0.5f, learning_rate = 0.;
    size_t batch_size = 5, max_epoch = 100;
    auto &training_set = training_collection->getTrainingSet();
    auto &eval_set = training_collection->getEvalSet();
    auto const &classes = training_collection->getClasses();

    size_t nclass = classes.size();
    math::Matrix<size_t> confusion(nclass, nclass);
    math::FloatMatrix target(nclass, 1);


    auto &optimizer = params->getOptimizer();

    std::vector<math::FloatMatrix> training_targets;
    tscl::logger("Building target matrices...");

    for (size_t i = 0; auto const &set : training_set) {
      target.fill(0);
      target(training_set.getLabel(i).getId(), 0) = 1.f;
      training_targets.push_back(target);
      i++;
    }

    tscl::logger("Training started with: {initial_learning_rate: " +
                         std::to_string(initial_learning_rate) +
                         ", batch_size: " + std::to_string(batch_size) +
                         ", Max epoch: " + std::to_string(max_epoch) + "}",
                 tscl::Log::Information);
    while (stracker.getEpoch() < max_epoch) {
      for (int i = 0; i < batch_size; i++) {
        optimizer.train(training_set.begin(), training_set.end(), training_targets.begin());
      }
      training_set.shuffle(std::random_device{}());

      confusion.fill(0);
      for (int i = 0; i < eval_set.size(); i++) {
        auto type = eval_set.getLabel(i).getId();
        auto res = model->predict(eval_set[i]);
        auto res_type = std::distance(res.begin(), std::max_element(res.begin(), res.end()));

        confusion(res_type, type)++;
      }

      auto stats = stracker.computeStats(confusion);
      tscl::logger("[Epoch " + std::to_string(stracker.getEpoch()) +
                           "]: avg_prec: " + std::to_string(stats.getAvgPrec()) +
                           ", avg_recall: " + std::to_string(stats.getAvgRecall()) +
                           ", avg_f1: " + std::to_string(stats.getAvgF1()),
                   tscl::Log::Information);
      stats.dumpToFiles();
      stracker.nextEpoch();
      optimizer.update();
    }
  }

  void CTController::printPostTrainingStats(CTracker &stracker) {
    auto &eval_set = training_collection->getEvalSet();
    auto &classes = training_collection->getClasses();
    size_t nclass = classes.size();
    math::Matrix<size_t> confusion(nclass, nclass);
    confusion.fill(0);

    std::stringstream ss;

    for (int i = 0; i < eval_set.size(); i++) {
      auto &type = eval_set.getLabel(i);
      auto res = model->predict(eval_set[i]);
      auto res_type = std::distance(res.begin(), std::max_element(res.begin(), res.end()));


      ss << "[Image: " << i << "]: (" << type << ") output :\n" << res;
      tscl::logger(ss.str(), tscl::Log::Information);
      ss.str("");
      confusion(res_type, type.getId())++;
    }
    auto stats = stracker.computeStats(confusion);
    // stats.classification_report(os, classes);
  }

}   // namespace control::classifier