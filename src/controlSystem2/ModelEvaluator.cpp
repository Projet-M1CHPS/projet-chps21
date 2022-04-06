#include "ModelEvaluator.hpp"

namespace control {

  ModelEvaluation ModelEvaluator::evaluate(const nnet::Model &model, const InputSet &input_set,
                                           size_t max_parallel_jobs) const {
    math::FloatMatrix confusion_matrix(input_set.getClassCount(), input_set.getClassCount());
    confusion_matrix.fill(0);

    for (auto &input : input_set) {
      long true_class = input.getClass();
      auto buf = model.predict(input.getData());
      size_t predicted_class = buf.imax();

      confusion_matrix(predicted_class, true_class)++;
    }
  }


  ModelEvolutionTracker::ModelEvolutionTracker(std::shared_ptr<ModelEvaluator> parent,
                                               const std::filesystem::path &output_path)
      : ModelEvaluatorDecorator(std::move(parent)), output_path(output_path) {}


  ModelEvaluation ModelEvolutionTracker::evaluate(const nnet::Model &model,
                                                  const InputSet &input_set,
                                                  size_t max_parallel_jobs) const {
    ModelEvaluation evaluation = parent->evaluate(model, input_set, max_parallel_jobs);
  }

  ModelVerboseEvaluator::ModelVerboseEvaluator(std::shared_ptr<ModelEvaluator> parent)
      : ModelEvaluatorDecorator(std::move(parent)) {}


  ModelEvaluation ModelVerboseEvaluator::evaluate(const nnet::Model &model,
                                                  const InputSet &input_set,
                                                  size_t max_parallel_jobs) const {
    ModelEvaluation evaluation = parent->evaluate(model, input_set, max_parallel_jobs);
  }

}   // namespace control