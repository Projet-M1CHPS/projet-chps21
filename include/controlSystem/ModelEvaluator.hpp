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


    ModelEvaluation(double avg_precision, std::vector<double> precision, double avg_recall,
                    std::vector<double> recall, double avg_f1score, std::vector<double> f1score)
        : avg_precision(avg_precision), precision(std::move(precision)), avg_recall(avg_recall),
          recall(std::move(recall)), avg_f1score(avg_f1score), f1score(std::move(f1score)) {}

    double getAvgPrecision() const { return avg_precision; }
    const std::vector<double> &getPrecision() const { return precision; }

    double getAvgRecall() const { return avg_recall; }

    const std::vector<double> &getRecall() const { return recall; }

    double getAvgF1score() const { return avg_f1score; }

    const std::vector<double> &getF1score() const { return f1score; }


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
    virtual ModelEvaluation evaluate(const nnet::Model &model, const InputSet &input_set) const;
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
      if (not this->parent) { throw std::invalid_argument("ModelEvaluatorDecorator: Decorator created with null parent"); }
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
                          std::filesystem::path output_path);

    /**
     * @brief Evaluate a model on an input set, and return the evaluation
     * @param model The model to evaluate
     * @param input_set The input set to run the evaluation on
     * @param max_parallel_jobs The maximum number of parallel jobs to run for the evaluation
     * @return
     */
    ModelEvaluation evaluate(const nnet::Model &model, const InputSet &input_set) const override;

  private:
    std::filesystem::path output_path;
  };

  /**
   * @brief Decorates a ModelEvaluator and automatically prints the evaluation to the console
   */
  class ModelVerboseEvaluator : public ModelEvaluatorDecorator {
  public:
    ModelVerboseEvaluator(std::shared_ptr<ModelEvaluator> parent);

    ModelEvaluation evaluate(const nnet::Model &model, const InputSet &input_set) const override;
  };

}   // namespace control
