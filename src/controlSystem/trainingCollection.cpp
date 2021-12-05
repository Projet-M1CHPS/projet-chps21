#include "controlSystem/trainingCollection.hpp"


namespace control {

  ClassifierTrainingCollection::ClassifierTrainingCollection()
      : class_labels(std::make_shared<std::set<ClassLabel>>()) {
    training_set = ClassifierInputSet(class_labels);
    eval_set = ClassifierInputSet(class_labels);
  }

  ClassifierTrainingCollection::ClassifierTrainingCollection(
          std::shared_ptr<std::set<ClassLabel>> classes)
      : class_labels(std::move(classes)) {
    training_set = ClassifierInputSet(class_labels);
    eval_set = ClassifierInputSet(class_labels);
  }


  void ClassifierTrainingCollection::shuffleTrainingSet(size_t seed) { training_set.shuffle(seed); }

  void ClassifierTrainingCollection::shuffleEvalSet(size_t seed) { eval_set.shuffle(seed); }

  void ClassifierTrainingCollection::shuffleSets(size_t seed) {
    shuffleTrainingSet(seed);
    shuffleEvalSet(seed);
  }

  void ClassifierTrainingCollection::unload() {
    training_set.unload();
    eval_set.unload();
  }

  std::ostream &operator<<(std::ostream &os, ClassifierTrainingCollection const &set) {
    os << "Classifier training set: " << std::endl;
    os << "\tTraining set contains " << set.training_set.size() << " elements" << std::endl;
    os << "\tEvaluation set contains " << set.eval_set.size() << " elements" << std::endl;

    os << "Classes: " << std::endl;
    for (auto const &label : *set.class_labels) { os << label << std::endl; }

    return os;
  }
}   // namespace control