#pragma once
#include "controlSystem/inputSet.hpp"

namespace control::classifier {

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

  class ClassifierInputSet : public InputSet<float> {
    friend std::ostream &operator<<(std::ostream &os, ClassifierInputSet const &set);

  public:
    ClassifierInputSet() = default;

    explicit ClassifierInputSet(std::shared_ptr<std::vector<ClassLabel>> labels)
        : class_labels(std::move(labels)) {}

    [[nodiscard]] ClassLabel const &getLabel(size_t index) const {
      if (index >= set_labels.size()) {
        throw std::out_of_range("ClassifierInputSet::getSetLabel Index out of range");
      }
      return *set_labels[index];
    }

    [[nodiscard]] std::vector<ClassLabel const *> const &getLabels() const { return set_labels; }
    [[nodiscard]] std::vector<ClassLabel> const &getClassLabels() const { return *class_labels; }


    void append(std::filesystem::path path, math::Matrix<float> &&mat) override;
    void append(std::filesystem::path path, ClassLabel const *label, math::Matrix<float> &&mat);

    void shuffle(size_t seed);

    void unload() override {
      InputSet::unload();
      set_labels.clear();
    }

  protected:
    std::vector<ClassLabel const *> set_labels;
    std::shared_ptr<std::vector<ClassLabel>> class_labels;
  };

}   // namespace control::classifier