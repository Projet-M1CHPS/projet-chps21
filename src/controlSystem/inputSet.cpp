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

  std::ostream &operator<<(std::ostream &os, const ClassLabel &label) {
    os << "\tclass_id: " << label.getId() << ", class_name: " << label.getName();
    return os;
  }
}   // namespace control
