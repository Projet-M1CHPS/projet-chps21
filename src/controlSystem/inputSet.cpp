#include <controlSystem/inputSet.hpp>
#include <utility>

namespace control {

  void InputSet::append(std::filesystem::path path, math::Matrix<float> &&mat) {
    inputs_path.push_back(std::move(path));
    inputs.push_back(std::move(mat));
  }

  void ClassifierInputSet::append(std::filesystem::path path, math::Matrix<float> &&mat) {
    InputSet::append(std::move(path), std::move(mat));
    set_labels.push_back(&ClassLabel::unknown);
  }

  void ClassifierInputSet::append(std::filesystem::path path, ClassLabel const *label,
                                  math::Matrix<float> &&mat) {
    if (label == nullptr) {
      append(std::move(path), std::move(mat));
      return;
    }

    if (not class_labels->contains(*label)) {
      throw std::runtime_error(
              "ClassifierInputSet::append: label doesn't belong to this classifier");
    }

    InputSet::append(std::move(path), std::move(mat));

    set_labels.push_back(label);
  }

  void ClassifierInputSet::shuffle(size_t seed) {
    std::mt19937_64 rng(seed);
    std::shuffle(inputs_path.begin(), inputs_path.end(), rng);

    rng.seed(seed);
    std::shuffle(inputs.begin(), inputs.end(), rng);

    rng.seed(seed);
    std::shuffle(set_labels.begin(), set_labels.end(), rng);
  }

  void ClassifierTrainingSet::shuffleTrainingSet(size_t seed) { training_set.shuffle(seed); }

  void ClassifierTrainingSet::shuffleEvalSet(size_t seed) { eval_set.shuffle(seed); }

  void ClassifierTrainingSet::shuffleSets(size_t seed) {
    shuffleTrainingSet(seed);
    shuffleEvalSet(seed);
  }

  void ClassifierTrainingSet::unload() {
    training_set.unload();
    eval_set.unload();
  }

  std::ostream &operator<<(std::ostream &os, InputSet const &set) {
    for (auto const &input : set.inputs_path) { os << "\t" << input << std::endl; }
    return os;
  }

  std::ostream &operator<<(std::ostream &os, const ClassifierInputSet &set) {
    for (size_t i = 0; i < set.size(); i++) {
      auto const &label = set.getLabel(i);
      os << "\t" << set.getPath(i) << "(class_id: " << label.getId()
         << ", class_name: " << label.getName() << std::endl;
    }
    return os;
  }

  std::ostream &operator<<(std::ostream &os, ClassifierTrainingSet const &set) {
    os << "Classifier training set: " << std::endl;
    os << "\tTraining set contains " << set.training_set.size() << " elements" << std::endl;
    os << "\tEvaluation set contains " << set.eval_set.size() << " elements" << std::endl;

    os << "Classes: " << std::endl;
    for (auto const &label : *set.class_labels) { os << label << std::endl; }

    return os;
  }

  ClassifierTrainingSet::ClassifierTrainingSet()
      : class_labels(std::make_shared<std::set<ClassLabel>>()) {
    training_set = ClassifierInputSet(class_labels);
    eval_set = ClassifierInputSet(class_labels);
  }
  ClassifierTrainingSet::ClassifierTrainingSet(std::shared_ptr<std::set<ClassLabel>> classes)
      : class_labels(std::move(classes)) {
    training_set = ClassifierInputSet(class_labels);
    eval_set = ClassifierInputSet(class_labels);
  }

  std::ostream &operator<<(std::ostream &os, const ClassLabel &label) {
    os << "\tclass_id: " << label.getId() << ", class_name: " << label.getName();
    return os;
  }
}   // namespace control
