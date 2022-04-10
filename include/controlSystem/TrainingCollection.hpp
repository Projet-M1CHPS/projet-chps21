#pragma once
#include "InputSet.hpp"

namespace control {

  /**
   * @brief A collection of samples that can be used for training a model. Provides a training set
   * and an evaluation set.
   */
  class TrainingCollection {
  public:
    friend class TrainingCollectionLoader;


    TrainingCollection(size_t input_width, size_t input_height)
        : training_set(input_width, input_height), eval_set(input_width, input_height) {}

    // Since copying the collection might be costly, we delete the default copy constructor and
    // assignment operator to prevent misuse
    TrainingCollection(const TrainingCollection &other) = delete;
    TrainingCollection &operator=(const TrainingCollection &other) = delete;

    TrainingCollection(TrainingCollection &&other) noexcept = default;
    TrainingCollection &operator=(TrainingCollection &&other) noexcept = default;

    InputSet &getTrainingSet() { return training_set; }

    const InputSet &getTrainingSet() const { return training_set; }

    InputSet &getEvaluationSet() { return eval_set; }

    const InputSet &getEvaluationSet() const { return eval_set; }

    size_t getClassCount() const { return eval_set.getClasses().size(); }

    void updateClasses(const std::vector<std::string> &class_names) {
      training_set.updateClasses(class_names);
      eval_set.updateClasses(class_names);
    }

    const std::vector<std::string> &getClassNames() const { return eval_set.getClasses(); }

    /**
     * @brief Alter the location of every input in both sets to create new tensors of the given
     * size. Residual elements will be grouped in a smaller tensors at the end of the lists Note
     * that this is a really costly operation, so it should be used with care.
     *
     * @details New tensors will be created with the given size, and the inputs will be moved
     * (Maintaining the same ordering) to fill them up. Older tensors are deleted, and the last
     * tensor might be smaller if the number of inputs is not a multiple of the new size.
     *
     * @param new_tensor_size The new size for the tensors
     */
    void alterTensors(size_t new_tensor_size) {
      training_set.alterTensors(new_tensor_size);
      eval_set.alterTensors(new_tensor_size);
    }

    /**
     * @brief Shuffle the inputs in both sets. Since moving each image individually
     * would be costly, this operation only shuffle the tensors.
     *
     * @param training_seed The random seed to be used for the training set
     * @param eval_seed The random seed to be used for the evaluation set
     */
    void shuffle(size_t training_seed, size_t eval_seed) {
      training_set.shuffle(training_seed);
      eval_set.shuffle(eval_seed);
    }

    /**
     * @brief Shuffle the inputs in both sets. Since moving each image individually
     * would be costly, this operation only shuffle the tensors.
     */
    void shuffle() {
      training_set.shuffle();
      eval_set.shuffle();
    }

  private:
    InputSet training_set;
    InputSet eval_set;
  };

}   // namespace control
