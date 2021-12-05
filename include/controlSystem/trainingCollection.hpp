#pragma once

#include "controlSystem/inputSet.hpp"

namespace control {

  class ClassifierTrainingCollection {
    friend std::ostream &operator<<(std::ostream &os, ClassifierTrainingCollection const &set);

  public:
    ClassifierTrainingCollection();
    ClassifierTrainingCollection(std::shared_ptr<std::set<ClassLabel>> classes);

    ClassifierTrainingCollection(ClassifierTrainingCollection const &other) = delete;

    ClassifierTrainingCollection(ClassifierTrainingCollection &&other) = default;
    ClassifierTrainingCollection &operator=(ClassifierTrainingCollection &&other) = default;

    [[nodiscard]] ClassifierInputSet &getTrainingSet() { return training_set; }
    [[nodiscard]] ClassifierInputSet const &getTrainingSet() const { return training_set; }

    [[nodiscard]] ClassifierInputSet &getEvalSet() { return eval_set; };
    [[nodiscard]] ClassifierInputSet const &getEvalSet() const { return eval_set; };

    [[nodiscard]] size_t categoryCount() const { return class_labels->size(); }

    [[nodiscard]] std::set<ClassLabel> &getLabels() { return *class_labels; }
    [[nodiscard]] std::set<ClassLabel> const &getLabels() const { return *class_labels; }

    template<typename iterator>
    void setCategories(iterator begin, iterator end) {
      class_labels->clear();
      class_labels->insert(class_labels->begin(), begin, end);
    }

    void shuffleTrainingSet(size_t seed);
    void shuffleEvalSet(size_t seed);
    void shuffleSets(size_t seed);

    [[nodiscard]] bool empty() const { return training_set.empty() && eval_set.empty(); }
    [[nodiscard]] size_t size() const { return training_set.size() + eval_set.size(); }

    void unload();

  private:
    ClassifierInputSet training_set, eval_set;
    std::shared_ptr<std::set<ClassLabel>> class_labels;
  };
}   // namespace control