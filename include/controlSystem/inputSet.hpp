#pragma once
#include "Matrix.hpp"
#include <filesystem>
#include <iostream>
#include <set>
#include <utility>
#include <vector>

namespace control {

  /** Store a set of input as matrices, and their associated input file
   *
   * @tparam real precision used for the matrices
   */
  template<typename real, typename = std::enable_if<std::is_floating_point_v<real>>>
  class InputSet {
    template<typename r>
    friend std::ostream &operator<<(std::ostream &os, InputSet<r> const &set);

  public:
    InputSet() = default;
    InputSet(InputSet const &other) = delete;

    InputSet(InputSet &&other) noexcept = default;
    InputSet &operator=(InputSet &&other) noexcept = default;

    /** Returns the path of an input file
     *
     * @param index
     * @return
     */
    [[nodiscard]] std::filesystem::path const &getPath(size_t index) const {
      if (index >= inputs_path.size()) {
        throw std::out_of_range("InputSet::getPath Index out of range");
      }
      return inputs_path[index];
    }

    /** Returns an input matrix
     *
     * @param index
     * @return
     */
    math::Matrix<real> const &operator[](size_t index) const {
      if (index >= inputs.size()) {
        throw std::out_of_range("InputSet::operator[] Index out of range");
      }
      return inputs[index];
    }

    /** Append an input to the set
     *
     * @param path
     * @param mat
     */
    virtual void append(std::filesystem::path path, math::Matrix<real> &&mat) {
      inputs_path.push_back(std::move(path));
      inputs.push_back(std::move(mat));
    }

    using Iterator = typename std::vector<math::Matrix<real>>::iterator;
    using ConstIterator = typename std::vector<math::Matrix<real>>::const_iterator;

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

    [[nodiscard]] ConstIterator begin() const { return inputs.begin(); }
    [[nodiscard]] ConstIterator end() const { return inputs.end(); }

    /** Returns true if the set is empty
     *
     * @return
     */
    [[nodiscard]] bool empty() const { return inputs.empty(); }

    /** Returns the number of element s in the set
     *
     * @return
     */
    [[nodiscard]] size_t size() const { return inputs.size(); }

    /** Detroy every element in the set
     *
     */
    virtual void unload() { *this = std::move(InputSet()); }


  protected:
    std::vector<std::filesystem::path> inputs_path;
    std::vector<math::Matrix<real>> inputs;
  };


  template<typename real>
  std::ostream &operator<<(std::ostream &os, InputSet<real> const &set) {
    for (auto const &input : set.inputs_path) { os << "\t" << input << std::endl; }
    return os;
  }

}   // namespace control