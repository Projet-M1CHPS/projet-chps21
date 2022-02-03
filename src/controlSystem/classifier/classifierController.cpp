
#include "classifierController.hpp"
#include "Network.hpp"
#include "classifierTracker.hpp"
#include "tscl.hpp"

namespace control::classifier {

  /** This function should not throw on error
   *
   */
  ControllerResult CTController::run() noexcept {
    tscl::logger("CTController: run started", tscl::Log::Debug);
    try {
      train();
    } catch (std::exception &e) {
      tscl::logger("CTController failed: " + std::string(e.what()), tscl::Log::Error);
      return {false, e.what()};
    }
    return {true, "Run successful"};
  }

  ControllerResult CTController::train() {
    if (not training_collection or not model or not optimizer) {
      throw std::runtime_error(
              "CTController: train called with missing training collection, model, or optimizer");
    }

    auto &classes = training_collection->getClasses();
    size_t nclass = classes.size();
    CTracker stracker(params.getOutputPath() / "eval", classes);

    trainingLoop(stracker);
    printPostTrainingStats(stracker);
    tscl::logger("Training finished", tscl::Log::Information);

    return {true, "Training finished"};
  }

  void CTController::trainingLoop(CTracker &stracker) {
    // Aliases for convenience
    size_t max_epoch = params.getMaxEpoch(), batch_size = params.getBatchSize();
    auto &training_set = training_collection->getTrainingSet();
    auto &eval_set = training_collection->getEvalSet();
    auto const &classes = training_collection->getClasses();

    // Used for computing stats
    size_t nclass = classes.size();
    CTracker training_tracker(params.getOutputPath() / "train", classes);
    math::Matrix<size_t> confusion(nclass, nclass);
    math::FloatMatrix target(nclass, 1);

    // BE CAREFUL to shuffle the training sets BEFORE building the target matrices
    training_collection->shuffleSets(std::random_device{}());

    // We build the target matrices
    std::vector<math::FloatMatrix> training_targets;
    tscl::logger("Building target matrices...");
    for (size_t i = 0; auto const &set : training_set) {
      target.fill(0);
      target(training_set.getLabel(i).getId(), 0) = 1.f;
      training_targets.push_back(target);
      i++;
    }

    tscl::logger("Training started", tscl::Log::Information);
    while (stracker.getEpoch() < max_epoch) {
      for (int i = 0; i < batch_size; i++) {
        // optimizer->optimize(training_set.getVector(), training_targets);
      }
      // Re-shuffle after each training batch
      // training_set.shuffle(std::random_device{}());

      confusion.fill(0);
      for (int i = 0; i < training_set.size(); i++) {
        auto type = training_set.getLabel(i).getId();
        // auto res = model->predict(training_set[i]);
        // auto res_type = std::distance(res.begin(), std::max_element(res.begin(), res.end()));

        // confusion(res_type, type)++;
      }
      auto tstats = training_tracker.computeStats(confusion);
      tscl::logger("[Epoch " + std::to_string(stracker.getEpoch()) +
                           "] Training: avg_prec: " + std::to_string(tstats.getAvgPrec()) +
                           ", avg_recall: " + std::to_string(tstats.getAvgRecall()) +
                           ", avg_f1: " + std::to_string(tstats.getAvgF1()),
                   tscl::Log::Information);
      tstats.dumpToFiles();
      training_tracker.nextEpoch();

      confusion.fill(0);
      for (int i = 0; i < eval_set.size(); i++) {
        auto type = eval_set.getLabel(i).getId();
        // auto res = model->predict(eval_set[i]);
        // auto res_type = std::distance(res.begin(), std::max_element(res.begin(), res.end()));

        // confusion(res_type, type)++;
      }

      auto stats = stracker.computeStats(confusion);
      tscl::logger("[Epoch " + std::to_string(stracker.getEpoch()) +
                           "] Eval : avg_prec: " + std::to_string(stats.getAvgPrec()) +
                           ", avg_recall: " + std::to_string(stats.getAvgRecall()) +
                           ", avg_f1: " + std::to_string(stats.getAvgF1()),
                   tscl::Log::Information);
      stats.dumpToFiles();
      stracker.nextEpoch();
      optimizer->update();
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
      // auto res = model->predict(eval_set[i]);
      // auto res_type = std::distance(res.begin(), std::max_element(res.begin(), res.end()));


      // ss << "[Image: " << i << ", " << type << "] output :\n" << res;
      tscl::logger(ss.str(), tscl::Log::Information);
      ss.str("");
      // confusion(res_type, type.getId())++;
    }
    auto stats = stracker.computeStats(confusion);
    // stats.classification_report(os, classes);
  }

}   // namespace control::classifier