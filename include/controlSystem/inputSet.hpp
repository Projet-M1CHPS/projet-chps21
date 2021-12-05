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

    std::filesystem::path const &getPath(size_t index) const {
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

  class ClassLabel {
    friend std::ostream &operator<<(std::ostream &os, ClassLabel const &label);

  public:
    ClassLabel(size_t id, std::string name) : id(id), name(std::move(name)) {}
    [[nodiscard]] size_t getId() const { return id; }
    [[nodiscard]] std::string const &getName() const { return name; }

    void setId(size_t id) { this->id = id; }
    void setName(std::string name) { this->name = std::move(name); }

    static ClassLabel const &getUnknown() {
      static ClassLabel unknown(0, "Unknown");
      return unknown;
    }

    inline static ClassLabel const &unknown = getUnknown();

    bool operator==(ClassLabel const &other) const { return id == other.id && name == other.name; }
    bool operator!=(ClassLabel const &other) const { return !(*this == other); }

    bool operator>(ClassLabel const &other) const { return id > other.id; }
    bool operator<(ClassLabel const &other) const { return id < other.id; }


  private:
    size_t id;
    std::string name;
  };

  class ClassifierInputSet : public InputSet {
    friend std::ostream &operator<<(std::ostream &os, ClassifierInputSet const &set);

  public:
    ClassifierInputSet() = default;

    explicit ClassifierInputSet(std::shared_ptr<std::set<ClassLabel>> labels)
        : class_labels(std::move(labels)) {}

    [[nodiscard]] ClassLabel const &getLabel(size_t index) const {
      if (index >= set_labels.size()) {
        throw std::out_of_range("ClassifierInputSet::getSetLabel Index out of range");
      }
      return *set_labels[index];
    }

    [[nodiscard]] std::vector<ClassLabel const *> const &getLabels() const { return set_labels; }
    [[nodiscard]] std::set<ClassLabel> const &getClassLabels() const { return *class_labels; }


    void append(std::filesystem::path path, math::Matrix<float> &&mat) override;
    void append(std::filesystem::path path, ClassLabel const *label, math::Matrix<float> &&mat);

    void shuffle(size_t seed);

    void unload() override {
      InputSet::unload();
      set_labels.clear();
    }

  protected:
    std::vector<ClassLabel const *> set_labels;
    std::shared_ptr<std::set<ClassLabel>> class_labels;
  };
}   // namespace control