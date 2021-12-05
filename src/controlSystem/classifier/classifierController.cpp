
#include "classifierController.hpp"
#include "classifierTracker.hpp"

namespace control::classifier {

  ControllerResult CTController::run(bool is_verbose, std::ostream *os) {
    auto &param = *params;
    RunPolicy policy = param.getRunPolicy();

    try {
      if (policy == RunPolicy::load) {
        // Tries to load a saved model, and throws an exception if it fails.
        ControllerResult res = load(is_verbose, os);
        if (not res) return res;
      } else if (policy == RunPolicy::create) {
        // Create a new model and erase any saved model
        ControllerResult res = create(is_verbose, os);
        if (not res) return res;
      } else if (policy == RunPolicy::tryLoad) {
        // Tries to load a saved model that fits the given parameters
        // Throws if the load fails or if the parameters don't match
        ControllerResult res = load(is_verbose, os);
        // FIXME: parameter verification
        return res;
      } else if (policy == RunPolicy::loadOrOverwrite) {
        // Tries to load a saved model that fits the given parameters
        // If it fails, creates a new one with the given parameters and overwrite any previously
        // saved model
        ControllerResult res = load(is_verbose, os);
        if (not res) {
          res = create(is_verbose, os);
          if (not res) return res;
        }
      }
      train(is_verbose, os);
    } catch (std::exception &e) { return {false, e.what()}; }
    return {true, "Run successful"};
  }

  ControllerResult CTController::load(bool is_verbose, std::ostream *os) {
    return {false, "not implemented"};
  }

  ControllerResult CTController::create(bool is_verbose, std::ostream *os) {
    loadTrainingSet(is_verbose, os);
    network = std::make_shared<nnet::NeuralNetwork<float>>();
    network->setLayersSize(params->getTopology());

    return {true, "Training set loaded"};
  }

  ControllerResult CTController::checkModel(bool is_verbose, std::ostream *os) {
    return {false, "not implemented"};
  }

  void CTController::loadTrainingSet(bool is_verbose, std::ostream *os) {
    try {
      if (not training_collection) {
        training_collection = params->getSetLoader().load(params->getInputPath(), is_verbose, os);
      }
    } catch (std::exception &e) {
      throw std::runtime_error("Error while loading training set: " + std::string(e.what()));
    }
  }


  ControllerResult CTController::train(bool is_verbose, std::ostream *os) {
    if (not training_collection or not network) {
      throw std::runtime_error("CTController: train called with missing training set or network");
    }

    double initial_learning_rate = 1.f, learning_rate = 0.;
    size_t batch_size = 10, max_epoch = 100;
    if (is_verbose)
      *os << std::setprecision(8)
          << "Training started with: {initial_learning_rate: " << initial_learning_rate
          << ", batch_size: " << batch_size << ", Max epoch: " << max_epoch << "}" << std::endl;

    training_collection->shuffleSets(std::random_device()());
    auto &training_set = training_collection->getTrainingSet();
    auto &eval_set = training_collection->getEvalSet();
    auto const &classes = training_collection->getLabels();

    CTracker stracker(params->getWorkingPath() / "output", classes.begin(), classes.end());
    math::Matrix<size_t> confusion(classes.size(), classes.size());
    math::FloatMatrix target(classes.size(), 1);

    while (stracker.getEpoch() < max_epoch) {
      learning_rate =
              initial_learning_rate * (1 / (1 + 0.9 * static_cast<double>(stracker.getEpoch())));
      for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < training_set.size(); j++) {
          auto type = training_set.getLabel(j).getId();
          target.fill(0);
          target(type, 0) = 1.f;

          network->train(training_set[j], target, learning_rate);
        }
      }

      confusion.fill(0);
      for (int i = 0; i < eval_set.size(); i++) {
        auto type = eval_set.getLabel(i).getId();
        auto res = network->predict(eval_set[i]);
        auto res_type = std::distance(res.begin(), std::max_element(res.begin(), res.end()));

        confusion(res_type, type)++;
      }

      auto stats = stracker.computeStats(confusion);
      if (is_verbose) *os << stats;
      stats.dumpToFiles();
      stracker.nextEpoch();
    }

    if (not is_verbose) return {true, "Training finished"};

    for (int i = 0; i < eval_set.size(); i++) {
      auto &type = eval_set.getLabel(i);
      auto res = network->predict(eval_set[i]);
      auto res_type = std::distance(res.begin(), std::max_element(res.begin(), res.end()));

      *os << "[Image: " << i << "]: type " << type << " output :\n" << res;
      confusion(res_type, type.getId())++;
    }
    *os << std::endl;
    auto stats = stracker.computeStats(confusion);
    stats.classification_report(*os, classes);
  }

}   // namespace control::classifier