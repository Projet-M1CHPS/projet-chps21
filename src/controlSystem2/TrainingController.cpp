#include "TrainingController.hpp"
#include "ModelEvaluator.hpp"
#include <chrono>

namespace chrono = std::chrono;

namespace control {

  namespace {
    std::vector<math::clFTensor> setupTargets(const InputSet &input_set) {
      std::vector<math::clFTensor> res;
      size_t nclass = input_set.getClasses().size();

      for (size_t sample_index = 0; auto &tensor : input_set.getTensors()) {
        size_t size = tensor.getZ();
        math::clFTensor buf(nclass, 1, size);

        for (size_t j = 0; j < size; j++) {
          math::FloatMatrix mat(nclass, 1);
          mat.fill(0.0f);
          mat(input_set.getClassOf(sample_index), 0) = 1.0f;
          buf.getMatrix(j) = mat;
          sample_index++;
        }
        res.push_back(std::move(buf));
      }
      return res;
    }

    chrono::milliseconds runEpoch(nnet::Optimizer &optimizer, const InputSet &input_set,
                                  const std::vector<math::clFTensor> &targets) {
      auto start = chrono::high_resolution_clock::now();
      optimizer.optimize(input_set.getTensors(), targets);
      auto end = chrono::high_resolution_clock::now();
      optimizer.update();
      return chrono::duration_cast<chrono::milliseconds>(end - start);
    }
  }   // namespace

  TrainingController::TrainingController(std::filesystem::path const &output_path,
                                         nnet::Model &model, nnet::Optimizer &optimizer,
                                         TrainingCollection &training_collection, size_t max_epoch,
                                         bool output_stats)
      : Controller(output_path), model(&model), optimizer(&optimizer),
        training_collection(&training_collection), max_epoch(max_epoch),
        is_outputting_stats(output_stats) {}

  ControllerResult TrainingController::run() {
    auto &training_set = training_collection->getTrainingSet();
    auto &eval_set = training_collection->getEvaluationSet();

    std::vector<math::clFTensor> targets = setupTargets(training_set);


    std::shared_ptr<ModelEvaluator> evaluator = std::make_shared<ModelEvaluator>();
    evaluator = std::make_shared<ModelVerboseEvaluator>(evaluator);

    for (size_t curr_epoch = 0; curr_epoch < max_epoch; curr_epoch++) {
      auto epoch_duration = runEpoch(*optimizer, training_set, targets);

      tscl::logger("Epoch " + std::to_string(curr_epoch) + " took " + std::to_string(epoch_duration.count()) + "ms",
                   tscl::Log::Information);

      //  Async evaluation to avoid downtime
      ModelEvaluation evaluation = evaluator->evaluate(*model, eval_set);
    }

    return {0, "Training completed"};
  }
}   // namespace control