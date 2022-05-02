#include "ModelEvaluator.hpp"
#include "image/Image.hpp"

#include <utility>

namespace control {

  namespace fs = std::filesystem;
  std::ostream &operator<<(std::ostream &os, const ModelEvaluation &me) {
    os << me.avg_f1score * 100.f << "% Average f1 score" << std::endl;
    os << "\t- " << me.avg_precision * 100.f << "% Average precision" << std::endl;
    os << "\t- " << me.avg_recall * 100.f << "% Average recall" << std::endl;
    return os;
  }


  ModelEvaluation ModelEvaluator::evaluate(const nnet::Model &model,
                                           const InputSet &input_set) const {
    size_t nclass = input_set.getClassCount();
    math::Matrix<size_t> confusion_matrix(input_set.getClassCount(), input_set.getClassCount());
    confusion_matrix.fill(0);

    // On huge input sets, the evaluation can take quite some time
    // No reason not to use OpenMP here
    // On huge input sets, the evaluation can take quite some time
    // No reason not to use OpenMP here
#pragma omp declare reduction (add_matrix : math::Matrix<size_t> : omp_out += omp_in ) initializer ( omp_priv(omp_orig) )
#pragma omp parallel for reduction(add_matrix: confusion_matrix) schedule(dynamic) default(none) shared(input_set, model) num_threads(1) // 4 max thread to avoid overloading the GPU
    for (auto &input : input_set) {
      long true_class = input.getClass();
      auto buf = model.predict(utils::cl_wrapper.getDefaultQueue(), input.getData());
      size_t predicted_class = buf.imax();

      confusion_matrix(predicted_class, true_class)++;
    }


    double avg_precision = 0, avg_recall = 0, avg_f1 = 0;
    std::vector<double> precisions(input_set.getClassCount()), recalls(input_set.getClassCount()),
            f1s(input_set.getClassCount());

#pragma omp parallel for reduction(+: avg_precision, avg_recall, avg_f1) default(none) \
        shared(nclass, confusion_matrix, precisions, recalls, f1s) schedule(dynamic)
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

    avg_precision /= (double) nclass;
    avg_recall /= (double) nclass;
    avg_f1 /= (double) nclass;
    return {avg_precision, avg_recall, avg_f1, precisions, recalls, f1s};
  }


  ModelEvolutionTracker::ModelEvolutionTracker(std::filesystem::path output_path,
                                               nnet::Model &model,
                                               const TrainingCollection &training_collection)
      : do_eval_on_test(false), model(&model), output_path(std::move(output_path)),
        training_collection(&training_collection) {
    if (not fs::exists(this->output_path)) fs::create_directories(this->output_path);
    setupStreams(eval_set_avg_output_streams, eval_set_output_streams, "eval");
  }

  void ModelEvolutionTracker::enableEvalOnTest() {
    setupStreams(training_set_avg_output_streams, training_set_output_streams, "test");
  }

  void ModelEvolutionTracker::setupStreams(ClassOutputStreams &avg_streams,
                                           std::vector<ClassOutputStreams> &streams,
                                           const fs::path &subdir) {
    // Destroy old streams if any
    streams.clear();
    streams.resize(training_collection->getClassCount());

    fs::path target_path = output_path / subdir;
    if (not fs::exists(target_path)) fs::create_directories(target_path);
    if (not fs::exists(target_path / "f1")) fs::create_directories(target_path / "f1");
    if (not fs::exists(target_path / "precision"))
      fs::create_directories(target_path / "precision");
    if (not fs::exists(target_path / "recall")) fs::create_directories(target_path / "recall");

    // If something fails, just throw an exception
    for (size_t i = 0; auto &c : training_collection->getClassNames()) {
      streams[i].s_f1_score.exceptions(std::ifstream::badbit);
      streams[i].s_f1_score.open(target_path / "f1" / (c + ".dat"));
      writeHeader(streams[i].s_f1_score, "t f1_score");

      streams[i].s_precision.exceptions(std::ifstream::badbit);
      streams[i].s_precision.open(target_path / "precision" / (c + ".dat"));
      writeHeader(streams[i].s_f1_score, "t precision");

      streams[i].s_recall.exceptions(std::ifstream::badbit);
      streams[i].s_recall.open(target_path / "recall" / (c + ".dat"));
      writeHeader(streams[i].s_f1_score, "t recall");
      i++;
    }


    avg_streams.s_precision.exceptions(std::ifstream::badbit);
    avg_streams.s_precision.open(target_path / "avg_precision.dat");
    writeHeader(avg_streams.s_precision, "t recall");

    avg_streams.s_recall.exceptions(std::ifstream::badbit);
    avg_streams.s_recall.open(target_path / "avg_recall.dat");
    writeHeader(avg_streams.s_recall, "t recall");
  }


  ModelEvaluation ModelEvolutionTracker::evaluate() {
    ModelEvaluation evaluation =
            evaluator.evaluate(*model, training_collection->getEvaluationSet());
    writeToStreams(evaluation, eval_set_avg_output_streams, eval_set_output_streams);

    if (do_eval_on_test) {
      evaluation = evaluator.evaluate(*model, training_collection->getTrainingSet());
      writeToStreams(evaluation, training_set_avg_output_streams, training_set_output_streams);
    }
    epoch++;
    return evaluation;
  }

  void ModelEvolutionTracker::writeHeader(std::ostream &stream, const std::string &label) {
    stream << "# " << label << std::endl;
  }

  void ModelEvolutionTracker::writeToStreams(ModelEvaluation &eval, ClassOutputStreams &avg_streams,
                                             std::vector<ClassOutputStreams> &streams) const {
    if (eval.f1score.size() != streams.size())
      throw std::runtime_error("ModelEvaluation::writeToStreams: size mismatch");

    avg_streams.s_f1_score << epoch << eval.avg_f1score << std::endl;
    avg_streams.s_precision << epoch << eval.avg_precision << std::endl;
    avg_streams.s_recall << epoch << eval.avg_recall << std::endl;

    for (size_t i = 0; i < eval.f1score.size(); i++) {
      streams[i].s_f1_score << epoch << eval.f1score[i] << std::endl;
      streams[i].s_precision << epoch << eval.precision[i] << std::endl;
      streams[i].s_recall << epoch << eval.recall[i] << std::endl;
    }
  }

}   // namespace control