#include "ModelEvaluator.hpp"
#include "image/Image.hpp"

#include <utility>

namespace control {

  ModelEvaluation ModelEvaluator::evaluate(const nnet::Model &model,
                                           const InputSet &input_set) const {
    size_t nclass = input_set.getClassCount();
    math::FloatMatrix confusion_matrix(input_set.getClassCount(), input_set.getClassCount());
    confusion_matrix.fill(0);

    for (auto &input : input_set) {
      long true_class = input.getClass();
      auto buf = model.predict(input.getData());
      auto fbuf = buf.toFloatMatrix();
      long predicted_class =
              std::distance(fbuf.begin(), std::max_element(fbuf.begin(), fbuf.end()));
      // size_t predicted_class = buf.imax();

      confusion_matrix(predicted_class, true_class)++;
    }
    std::cout << confusion_matrix << std::endl << std::endl;

    double avg_precision = 0, avg_recall = 0, avg_f1 = 0;
    std::vector<double> precisions(input_set.getClassCount()), recalls(input_set.getClassCount()),
            f1s(input_set.getClassCount());

    for (size_t i = 0; i < nclass; i++) {
      size_t sum = 0;
      double recall = 0, precision = 0, f1 = 0;

      // Compute the precision
      for (size_t j = 0; j < nclass; j++) { sum += confusion_matrix(i, j); }
      if (sum != 0)
        precision = static_cast<double>(confusion_matrix(i, i)) / static_cast<double>(sum);


      // Compute the recall
      sum = 0;
      for (size_t j = 0; j < nclass; j++) { sum += confusion_matrix(j, i); }
      if (sum != 0) recall = static_cast<double>(confusion_matrix(i, i)) / static_cast<double>(sum);


      if (precision != 0 && recall != 0)
        // The f1 score is the harmonic mean of both the recall and the precision
        f1 = 2 * precision * recall / (precision + recall);
      precisions[i] = precision;
      recalls[i] = recall;
      f1s[i] = f1;

      avg_precision += precision;
      avg_recall += recall;
      avg_f1 += f1;
    }

    avg_precision /= nclass;
    avg_recall /= nclass;
    avg_f1 /= nclass;
    return {avg_precision, precisions, avg_recall, recalls, avg_f1, f1s};
  }


  ModelEvolutionTracker::ModelEvolutionTracker(std::shared_ptr<ModelEvaluator> parent,
                                               std::filesystem::path output_path)
      : ModelEvaluatorDecorator(std::move(parent)), output_path(std::move(output_path)) {}


  ModelEvaluation ModelEvolutionTracker::evaluate(const nnet::Model &model,
                                                  const InputSet &input_set) const {
    ModelEvaluation evaluation = parent->evaluate(model, input_set);

    return evaluation;
  }

  ModelVerboseEvaluator::ModelVerboseEvaluator(std::shared_ptr<ModelEvaluator> parent)
      : ModelEvaluatorDecorator(std::move(parent)) {}


  ModelEvaluation ModelVerboseEvaluator::evaluate(const nnet::Model &model,
                                                  const InputSet &input_set) const {
    ModelEvaluation evaluation = parent->evaluate(model, input_set);
    std::stringstream ss;
    ss << "Model evaluation: Average f1 score: " << evaluation.getAvgF1score() << "\n"
       << "  Average precision: " << evaluation.getAvgPrecision() << "\n"
       << "  Average recall: " << evaluation.getAvgRecall();
    tscl::logger(ss.str(), tscl::Log::Information);

    return evaluation;
  }

}   // namespace control