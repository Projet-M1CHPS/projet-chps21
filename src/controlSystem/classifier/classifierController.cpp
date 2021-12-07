
#include "classifierController.hpp"
#include "classifierTracker.hpp"
#include "neuralNetwork/NeuralNetwork.hpp"
#include "neuralNetwork/Optimizer.hpp"

namespace control::classifier {

  /** This function should not throw on error
   *
   */
  ControllerResult CTController::run(bool is_verbose, std::ostream *os) noexcept {
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
    auto topology = params->getTopology();

    topology.push_back(training_collection->classCount());
    network->setLayersSize(topology);
    network->randomizeSynapses();

    network->setActivationFunction(af::ActivationFunctionType::sigmoid);
    return {true, "Training set loaded"};
  }

  ControllerResult CTController::checkModel(bool is_verbose, std::ostream *os) {
    return {false, "not implemented"};
  }

  void CTController::loadTrainingSet(bool is_verbose, std::ostream *os) {
    try {
      if (not training_collection) {
        training_collection = params->getSetLoader().load(params->getInputPath(), is_verbose, os);
        training_collection->shuffleSets(std::random_device()());
      }
    } catch (std::exception &e) {
      throw std::runtime_error("Error while loading training set: " + std::string(e.what()));
    }
  }


  ControllerResult CTController::train(bool is_verbose, std::ostream *os) {
    if (not training_collection or not network) {
      throw std::runtime_error("CTController: train called with missing training set or network");
    }

    auto &classes = training_collection->getClasses();
    size_t nclass = training_collection->classCount();
    CTracker stracker(params->getWorkingPath() / "output", classes.begin(), classes.end());

    trainingLoop(is_verbose, os, stracker);
    if (is_verbose) printPostTrainingStats(*os, stracker);

    return {true, "Training finished"};
  }

  void CTController::trainingLoop(bool is_verbose, std::ostream *os, CTracker &stracker) {
    double initial_learning_rate = 0.5f, learning_rate = 0.;
    size_t batch_size = 5, max_epoch = 400;
    if (is_verbose)
      *os << std::setprecision(16)
          << "Training started with: {initial_learning_rate: " << initial_learning_rate
          << ", batch_size: " << batch_size << ", Max epoch: " << max_epoch << "}" << std::endl;

    auto &training_set = training_collection->getTrainingSet();
    auto &eval_set = training_collection->getEvalSet();
    auto const &classes = training_collection->getClasses();

    size_t nclass = classes.size();
    math::Matrix<size_t> confusion(nclass, nclass);
    math::FloatMatrix target(nclass, 1);

    nnet::DecayTrainingMethod<float> decay(initial_learning_rate, 0.9f);
    nnet::RPropPTrainingMethod<float> rprop(network->getLayersSize());
    nnet::MLPOptimizer<float> optimizer(network.get(), &rprop);

    while (stracker.getEpoch() < max_epoch) {
      learning_rate =
              initial_learning_rate * (1 / (1 + 0.5 * static_cast<double>(stracker.getEpoch())));
      for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < training_set.size(); j++) {
          auto type = training_set.getLabel(j).getId();
          target.fill(0);
          target(type, 0) = 1.f;


          optimizer.train(training_set[j], target);
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
  }

  void CTController::printPostTrainingStats(std::ostream &os, CTracker &stracker) {
    auto &eval_set = training_collection->getEvalSet();
    auto &classes = training_collection->getClasses();
    size_t nclass = classes.size();
    math::Matrix<size_t> confusion(nclass, nclass);

    for (int i = 0; i < eval_set.size(); i++) {
      auto &type = eval_set.getLabel(i);
      auto res = network->predict(eval_set[i]);
      auto res_type = std::distance(res.begin(), std::max_element(res.begin(), res.end()));

      os << "[Image: " << i << "]: (" << type << ") output :\n" << res;
      confusion(res_type, type.getId())++;
    }
    os << std::endl;
    auto stats = stracker.computeStats(confusion);
    stats.classification_report(os, classes);
  }

}   // namespace control::classifier