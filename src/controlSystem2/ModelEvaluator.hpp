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

namespace control {

  class ModelEvaluation {
  public:
    ModelEvaluation() : avg_precision(0), avg_f1score(0), avg_recall(0) {}


    ModelEvaluation(double avg_accuracy, std::vector<double> accuracy, double avg_recall,
                    std::vector<double> recall, double avg_f1score, std::vector<double> f1score);

    double getAvgPrecision() const;
    const std::vector<double> &getPrecision() const;

    double getAvgRecall() const;
    const std::vector<double> &getRecall() const;

    double getAvgF1score() const;
    const std::vector<double> &getF1score() const;


  private:
    double avg_precision;
    std::vector<double> precision;

    double avg_recall;
    std::vector<double> recall;

    double avg_f1score;
    std::vector<double> f1score;
  };

  /**
   * @brief Base class to evaluate a model on an input set
   */
  class ModelEvaluator {
  public:
    /**
     * @brief Evaluate a model on an input set, and return the evaluation
     * @param model The model to evaluate
     * @param input_set The input set to run the evaluation on
     * @param max_parallel_jobs The maximum number of parallel jobs to run for the evaluation
     * @return
     */
    virtual ModelEvaluation evaluate(const nnet::Model &model, const InputSet &input_set,
                                     size_t max_parallel_jobs) const;
  };

  /**
   * @brief Decorates a model evaluator to provide additional functionality
   */
  class ModelEvaluatorDecorator : public ModelEvaluator {
  public:
    /**
     * @brief Construct a new Model Evaluator Decorator object. If parent is nullptr, throws
     * @param parent
     */
    explicit ModelEvaluatorDecorator(std::shared_ptr<ModelEvaluator> parent)
        : parent(std::move(parent)) {
      if (not parent) { throw std::invalid_argument("Decorator created with null parent"); }
    }

  protected:
    std::shared_ptr<ModelEvaluator> parent;
  };

  /**
   * @brief Decorates a ModelEvaluator and provides the ability to save the model evolution over
   * time in data files
   */
  class ModelEvolutionTracker : public ModelEvaluatorDecorator {
  public:
    /**
     * @brief Build a new tracker to track the given ModelEvaluator over time
     * @param parent The parent to track
     * @param output_path The path to save the evolution to
     */
    ModelEvolutionTracker(std::shared_ptr<ModelEvaluator> parent,
                          const std::filesystem::path &output_path);

    /**
     * @brief Evaluate a model on an input set, and return the evaluation
     * @param model The model to evaluate
     * @param input_set The input set to run the evaluation on
     * @param max_parallel_jobs The maximum number of parallel jobs to run for the evaluation
     * @return
     */
    ModelEvaluation evaluate(const nnet::Model &model, const InputSet &input_set,
                             size_t max_parallel_jobs) const override;

  private:
    std::filesystem::path output_path;
  };

  /**
   * @brief Decorates a ModelEvaluator and automatically prints the evaluation to the console
   */
  class ModelVerboseEvaluator : public ModelEvaluatorDecorator {
  public:
    ModelVerboseEvaluator(std::shared_ptr<ModelEvaluator> parent);

    ModelEvaluation evaluate(const nnet::Model &model, const InputSet &input_set,
                             size_t max_parallel_jobs) const override;
  };

}   // namespace control
