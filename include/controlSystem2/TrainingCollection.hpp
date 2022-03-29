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

    // Since copying the collection might be costly, we delete the default copy constructor and
    // assignment operator to prevent misuse
    TrainingCollection(const TrainingCollection &other) = delete;
    TrainingCollection &operator=(const TrainingCollection &other) = delete;

    TrainingCollection &operator=(TrainingCollection &&other) = default;

    InputSet &getTrainingSet() { return training_set; }

    const InputSet &getTrainingSet() const { return training_set; }

    InputSet &getEvaluationSet() { return eval_set; }

    const InputSet &getEvaluationSet() const { return eval_set; }

    size_t getClassCount() const { return class_names.size(); }

    void updateClassNames(const std::vector<std::string> &class_names);

    std::string getClassName(size_t class_index) const {
      if (class_index >= class_names.size()) {
        throw std::out_of_range("class index out of range");
      }
      return class_names[class_index];
    }

    const std::vector<std::string> &getClassNames() const { return class_names; }

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
     * @return
     */
    size_t alterTensors(size_t new_tensor_size);

    /**
     * @brief Shuffle the inputs in both sets. Since moving each image individually
     * would be costly, this operation only shuffle the tensors.
     *
     * @param training_seed The random seed to be used for the training set
     * @param eval_seed The random seed to be used for the evaluation set
     */
    void shuffle(size_t training_seed, size_t eval_seed);

    /**
     * @brief Shuffle the inputs in both sets. Since moving each image individually
     * would be costly, this operation only shuffle the tensors.
     */
    void shuffle();

  private:
    InputSet training_set;
    InputSet eval_set;
    std::vector<std::string> class_names;
  };

}   // namespace control
