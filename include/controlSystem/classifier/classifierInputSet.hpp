#pragma once
#include "classifierClass.hpp"
#include "controlSystem/inputSet.hpp"

namespace control::classifier {

  /** @brief Input set specialization for classifiers
   *
   */
  class ClassifierTrainingSet : public InputSet {
  public:
    ClassifierTrainingSet() = default;

    /** Returns the nth input matrix
     *
     * Defined here for inlining purposes
     *
     * @param index Index of the matrix to retrieve
     * @return
     */
    [[nodiscard]] ClassLabel const &getLabel(size_t index) const {
      if (index >= set_labels.size()) {
        throw std::out_of_range("ClassifierInputSet::getLabel: Index out of range");
      }
      return *set_labels[index];
    }

    /** Appends a new input matrix to the set
     *
     * @param input_id The unique id of the input matrix
     * @param label Class label of the matrix
     * @param mat Input matrix
     */
    void append(size_t input_id, ClassLabel const *label, math::clMatrix &&mat);

    /** Appends a new input matrix to the set
     *
     * @param input_id The unique id of the input matrix
     * @param label Class label of the matrix
     * @param mat Input matrix
     */
    void append(size_t input_id, ClassLabel const *label, const math::clMatrix &mat);

    /** Shuffles the set with the given seed
     *
     * @param seed
     */
    void shuffle(size_t seed);

    /** Unloads the set, freeing all the memory
     *
     */
    void clear() override;

  protected:
    void print(std::ostream &os) const override;

    /* The set of the matrices ids
     *
     * This is used for retrieving metadata about the inputs, such as the origin file
     */
    std::vector<size_t> inputs_id;

    /** Every matrix is associated with a class label
     *
     */
    std::vector<ClassLabel const *> set_labels;
  };

}   // namespace control::classifier