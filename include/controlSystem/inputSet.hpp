#pragma once
#include "Matrix.hpp"
#include <filesystem>
#include <iostream>
#include <ranges>
#include <set>
#include <utility>
#include <vector>

namespace control {

  class InputSet {
    friend std::ostream &operator<<(std::ostream &os, InputSet const &set);

  public:
    InputSet() = default;
    InputSet(InputSet const &other) = delete;

    InputSet(InputSet &&other) = default;
    InputSet &operator=(InputSet &&other) = default;

    [[nodiscard]] std::filesystem::path const &getPath(size_t index) const {
      if (index >= inputs_path.size()) {
        throw std::out_of_range("InputSet::getPath Index out of range");
      }
      return inputs_path[index];
    }

    math::Matrix<float> const &operator[](size_t index) const {
      if (index >= inputs.size()) {
        throw std::out_of_range("InputSet::operator[] Index out of range");
      }
      return inputs[index];
    }

    virtual void append(std::filesystem::path path, math::Matrix<float> &&mat);

    using Iterator = std::vector<math::FloatMatrix>::iterator;
    using ConstIterator = std::vector<math::FloatMatrix>::const_iterator;

    [[nodiscard]] Iterator begin() { return inputs.begin(); }
    [[nodiscard]] Iterator end() { return inputs.end(); }

    [[nodiscard]] ConstIterator begin() const { return inputs.begin(); }
    [[nodiscard]] ConstIterator end() const { return inputs.end(); }

    [[nodiscard]] bool empty() const { return inputs.empty(); }
    [[nodiscard]] size_t size() const { return inputs.size(); }
    virtual void unload() { *this = std::move(InputSet()); }


  protected:
    std::vector<std::filesystem::path> inputs_path;
    std::vector<math::FloatMatrix> inputs;
  };
}   // namespace control