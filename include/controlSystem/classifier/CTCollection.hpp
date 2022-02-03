#pragma once

#include "classifierInputSet.hpp"

namespace control::classifier {

  /** @brief A collection used for training a classifier Model
   *
   * Pairs a set of inputs, one for training and one for evaluation
   * Every input (in both sets) is uniquely identified by its id, which can be used for metadata
   * retrieving
   *
   */
  class CTCollection {
    friend std::ostream &operator<<(std::ostream &os, CTCollection const &set);

  public:
    /** Create a collection with a given label list
     *
     *
     * @param classes
     */
    explicit CTCollection(std::shared_ptr<CClassLabelSet> classes);

    /** Collection can be really huge, so we delete the copy operators for safety
     *
     * @param other
     */
    // FIXME Add a copy method
    CTCollection(CTCollection const &other) = delete;

    CTCollection(CTCollection &&other) = default;
    CTCollection &operator=(CTCollection &&other) = default;

    /** Return the training set
     *
     * @param set
     */
    [[nodiscard]] ClassifierTrainingSet &getTrainingSet() { return training_set; }
    [[nodiscard]] ClassifierTrainingSet const &getTrainingSet() const { return training_set; }

    /** Returns the evaluation set
     *
     * @return
     */
    [[nodiscard]] ClassifierTrainingSet &getEvalSet() { return eval_set; };
    [[nodiscard]] ClassifierTrainingSet const &getEvalSet() const { return eval_set; };

    /** Returns the number of classes in this collection
     *
     * @return
     */
    [[nodiscard]] size_t getClassCount() const { return class_list->size(); }

    /** Return the classes used in this collection
     *
     * @return
     */
    [[nodiscard]] CClassLabelSet const &getClasses() const { return *class_list; }

    /** Randomly shuffle the training set
     *
     * Training should be done on a shuffled set for a uniform learning that is not biased by the
     * inputs order
     *
     * @param seed
     */
    void shuffleTrainingSet(size_t seed) { training_set.shuffle(seed); }

    /* Randomly shuffle the eval set
     *
     */
    void shuffleEvalSet(size_t seed) { eval_set.shuffle(seed); }

    /** Randomly shuffles both sets with a seed
     * The same seed is used for both shuffling
     *
     * @param seed
     */
    void shuffleSets(size_t seed) {
      shuffleTrainingSet(seed);
      shuffleEvalSet(seed);
    }

    /** Returns true if both sets are empty
     *
     * @return
     */
    [[nodiscard]] bool empty() const { return training_set.empty() && eval_set.empty(); }

    /** Returns the sum of the sizes of both sets
     *
     * @return
     */
    [[nodiscard]] size_t size() const { return training_set.size() + eval_set.size(); }

    /** Unload both sets, freeing any used memory
     *
     */
    void unload() {
      training_set.clear();
      eval_set.clear();
      class_list = nullptr;
    }

  private:
    ClassifierTrainingSet training_set, eval_set;
    std::shared_ptr<CClassLabelSet> class_list;
  };
}   // namespace control::classifier