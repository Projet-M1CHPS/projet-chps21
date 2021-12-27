#pragma once
#include "Matrix.hpp"
#include <filesystem>
#include <iostream>
#include <set>
#include <utility>
#include <vector>

namespace control {

  /** @brief Stores a set of matrices to be fed to a model
   *
   * This base class only stores matrices, and has no additional information about the input paths
   * and is agnostic to the model used.
   *
   * Specialization may store additional information such as the input paths, or the inputs class
   * for classifiers
   *
   */
  class InputSet {
    friend std::ostream &operator<<(std::ostream &os, InputSet const &set);

  public:
    InputSet() = default;

    /** Sets can get really huge, so we destroy the copy constructor by precaution
     * FIXME: Add a copy() method
     *
     * @param other
     */
    InputSet(InputSet const &other) = delete;

    InputSet(InputSet &&other) noexcept = default;
    InputSet &operator=(InputSet &&other) noexcept = default;

    /** Returns an input matrix
     *
     * Defined here to allow inlining
     *
     * @param index
     * @return
     */
    math::Matrix<float> const &operator[](size_t index) const {
      if (index >= inputs.size()) {
        throw std::out_of_range("InputSet::operator[] Index out of range");
      }
      return inputs[index];
    }

    // This totally breaks encapsulation, but is the simplest way I found to pass the matrices to
    // the optimizer
    /** Return the internal storage used by the set
     *
     * @return
     */
    [[nodiscard]] const std::vector<math::Matrix<float>> &getVector() const { return inputs; }

    using Iterator = typename std::vector<math::Matrix<float>>::iterator;
    using ConstIterator = typename std::vector<math::Matrix<float>>::const_iterator;

    /** Returns an iterator to the first input matrix
     *
     * @return
     */
    [[nodiscard]] Iterator begin() { return inputs.begin(); }

    /** Returns an iterator to the end of the input matrices
     *
     * @return
     */
    [[nodiscard]] Iterator end() { return inputs.end(); }

    /** Used for const for-range loops
     *
     * @return
     */
    [[nodiscard]] ConstIterator begin() const { return inputs.begin(); }
    [[nodiscard]] ConstIterator cbegin() const { return inputs.cbegin(); }

    [[nodiscard]] ConstIterator end() const { return inputs.end(); }
    [[nodiscard]] ConstIterator cend() const { return inputs.cend(); }


    /** Returns true if the set is empty
     *
     * @return
     */
    [[nodiscard]] bool empty() const { return inputs.empty(); }

    /** Returns the number of element stored
     *
     * @return
     */
    [[nodiscard]] size_t size() const { return inputs.size(); }

    /** Free every element in the set
     *
     */
    virtual void clear() { inputs.clear(); }

  protected:
    virtual void print(std::ostream &os) const;

    std::vector<math::Matrix<float>> inputs;
  };

}   // namespace control