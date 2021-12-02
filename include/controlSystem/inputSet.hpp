#pragma once
#include "Matrix.hpp"
#include <filesystem>
#include <iostream>
#include <vector>

namespace control {

  class InputSet {
    friend std::ostream &operator<<(std::ostream &os, InputSet const &set);

  public:
    InputSet() = default;
    InputSet(InputSet const &other) = delete;

    InputSet(InputSet &&other) = default;
    InputSet &operator=(InputSet &&other) = default;

    std::filesystem::path const &getPath(size_t index) {
      if (index >= inputs_path.size()) {
        throw std::out_of_range("InputSet::getPath Index out of range");
      }
    }

    math::Matrix<float> const &operator[](size_t index) const {
      if (index >= inputs.size()) {
        throw std::out_of_range("InputSet::operator[] Index out of range");
      }
      return inputs[index];
    }

    void append(std::filesystem::path path, math::Matrix<float> &&mat);

    [[nodiscard]] bool empty() const { return inputs.empty(); }
    [[nodiscard]] size_t size() const { return inputs.size(); }
    void unload() { *this = std::move(InputSet()); }

  private:
    std::vector<std::filesystem::path> inputs_path;
    std::vector<math::Matrix<float>> inputs;
  };

  std::ostream &operator<<(std::ostream &os, InputSet const &set);

  class TrainingSet {
    friend std::ostream &operator<<(std::ostream &os, TrainingSet const &set);

  public:
    TrainingSet() = default;

    TrainingSet(TrainingSet const &other) = delete;

    TrainingSet(TrainingSet &&other) = default;
    TrainingSet &operator=(TrainingSet &&other) = default;

    [[nodiscard]] std::filesystem::path const &getTrainingPath(size_t index) const {
      if (index > training_set_files.size())
        throw std::out_of_range("TrainingSet::getTrainingPath: index out of range");
      return training_set_files[index];
    }
    [[nodiscard]] std::filesystem::path const &getEvalPath(size_t index) const {
      if (index > eval_set_files.size())
        throw std::out_of_range("TrainingSet::getEvalPath: index out of range");
      return eval_set_files[index];
    }

    [[nodiscard]] math::Matrix<float> const &getTrainingMat(size_t index) const {
      if (index > training_set.size())
        throw std::out_of_range("TrainingSet::getTrainingMat: index out of range");
      return training_set[index];
    }

    [[nodiscard]] math::Matrix<float> const &getEvalMat(size_t index) const {
      if (index > eval_set.size())
        throw std::out_of_range("TrainingSet::getEvalMat: index out of range");
      return eval_set[index];
    }

    [[nodiscard]] size_t getTrainingCategory(size_t index) const {
      if (index > training_set_categories.size())
        throw std::out_of_range("TrainingSet::getTrainingCategory: index out of range");
      return training_set_categories[index];
    }
    [[nodiscard]] size_t getEvalCategory(size_t index) const {
      if (index > eval_set_categories.size())
        throw std::out_of_range("TrainingSet::getEvalCategory: index out of range");
      return eval_set_categories[index];
    }

    [[nodiscard]] std::string const &getCategory(size_t index) const {
      if (index > categories.size())
        throw std::out_of_range("TrainingSet::getCategory: index out of range");
      return categories[index];
    }
    [[nodiscard]] std::vector<std::string> const &getCategories() const { return categories; }

    template<typename iterator>
    void setCategories(iterator begin, iterator end) {
      categories.clear();

      categories.reserve(std::distance(begin, end));
      categories.insert(categories.begin(), begin, end);
    }

    void appendToTrainingSet(std::filesystem::path path, size_t category,
                             math::Matrix<float> &&mat);
    void appendToEvalSet(std::filesystem::path path, size_t category, math::Matrix<float> &&mat);

    void shuffleTrainingSet(size_t seed);
    void shuffleEvalSet(size_t seed);
    void shuffleSets(size_t seed);

    [[nodiscard]] bool empty() const { return training_set.empty() && eval_set.empty(); }
    [[nodiscard]] size_t size() const { return training_set.size() + eval_set.size(); }

    [[nodiscard]] size_t trainingSetSize() const;
    [[nodiscard]] size_t evalSetSize() const;

    void unload();

  private:
    std::vector<std::filesystem::path> training_set_files;
    std::vector<std::filesystem::path> eval_set_files;

    std::vector<math::Matrix<float>> training_set;
    std::vector<math::Matrix<float>> eval_set;

    std::vector<size_t> training_set_categories;
    std::vector<size_t> eval_set_categories;

    std::vector<std::string> categories;
  };

  std::ostream &operator<<(std::ostream &os, TrainingSet const &);

}   // namespace control