/**
 * @brief Provides an interface for evaluating the model on an input set and compute some
 * statistics, such as f1_score, accuracy, etc.
 *
 * The following file implements a decorator pattern.
 */
#pragma once
#include <utility>

#include "InputSet.hpp"
#include "Model.hpp"
#include "TrainingCollection.hpp"
#include <fstream>

namespace control {

  /**
   * @brief Structs containing the results of the evaluation.
   * @details The results of the evaluation is raw data without semantic attached to it, hence we
   * use a struct to store the results. This avoids the need to pass the results around, or the
   * needs for getters / setters that would break encapsulation anyway.
   */
  struct ModelEvaluation {
    friend std::ostream &operator<<(std::ostream &os, const ModelEvaluation &me);

    double avg_precision = 0;
    double avg_recall = 0;
    double avg_f1score = 0;

    std::vector<double> precision;
    std::vector<double> recall;
    std::vector<double> f1score;
  };

  /**
   * @brief Base class to evaluate a model on an input set
   */
  class ModelEvaluator {
  public:
    virtual ~ModelEvaluator() = default;

    /**
     * @brief Evaluate a model on an input set, and return the evaluation
     * @param model The model to evaluate
     * @param input_set The input set to run the evaluation on
     * @return
     */
    virtual ModelEvaluation evaluate(const nnet::Model &model, const InputSet &input) const;
  };


  /**
   * @brief Decorates a ModelEvaluator and provides the ability to save the model evolution over
   * time in data files
   */
  class ModelEvolutionTracker {
  public:
    /**
     * @brief Build a new tracker to track the given ModelEvaluator over time
     * @param parent The parent to track
     * @param output_path The path to save the evolution to
     */
    ModelEvolutionTracker(std::filesystem::path output_path, nnet::Model &model,
                          const TrainingCollection &training_collection);

    /**
     * @brief Evaluate a model on an input set, and return the evaluation
     * @param model The model to evaluate
     * @param input_set The input set to run the evaluation on
     * @param max_parallel_jobs The maximum number of parallel jobs to run for the evaluation
     * @return
     */
    ModelEvaluation evaluate();

    void enableEvalOnTest();

  private:
    bool do_eval_on_test;
    size_t epoch = 0;

    nnet::Model *model;

    ModelEvaluator evaluator;
    std::filesystem::path output_path;

    const TrainingCollection *training_collection;

    // Helper struct to store the various streams of each class
    // Declared as private to prevent accidental access
    struct ClassOutputStreams {
      std::ofstream s_f1_score;
      std::ofstream s_precision;
      std::ofstream s_recall;
    };

    void setupStreams(ClassOutputStreams &avg_streams, std::vector<ClassOutputStreams> &streams,
                      const std::filesystem::path &subdir);

    std::vector<ClassOutputStreams> eval_set_output_streams;
    std::vector<ClassOutputStreams> training_set_output_streams;
    ClassOutputStreams eval_set_avg_output_streams;
    ClassOutputStreams training_set_avg_output_streams;

    static void writeHeader(std::ostream &stream, const std::string &label);
    void writeToStreams(ModelEvaluation &eval, ClassOutputStreams &avg_streams,
                        std::vector<ClassOutputStreams> &streams) const;
  };
}   // namespace control
