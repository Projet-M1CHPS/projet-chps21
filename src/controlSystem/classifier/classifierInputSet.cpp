#include "classifierInputSet.hpp"

namespace control::classifier {

  void ClassifierTrainingSet::append(size_t input_id, ClassLabel const *label,
                                     math::Matrix<float> &&mat) {
    if (label == nullptr) {
      throw std::invalid_argument("ClassifierTrainingSet: label is nullptr");
    }
    inputs.push_back(std::move(mat));
    inputs_id.push_back(input_id);
    set_labels.push_back(label);
  }

  void ClassifierTrainingSet::append(size_t input_id, const ClassLabel *label,
                                     const math::Matrix<float> &mat) {
    if (label == nullptr) {
      throw std::invalid_argument("ClassifierTrainingSet: label is nullptr");
    }
    inputs.push_back(mat);
    inputs_id.push_back(input_id);
    set_labels.push_back(label);
  }


  void ClassifierTrainingSet::shuffle(size_t seed) {
    std::mt19937_64 rng(seed);
    std::shuffle(inputs_id.begin(), inputs_id.end(), rng);

    rng.seed(seed);
    std::shuffle(inputs.begin(), inputs.end(), rng);

    rng.seed(seed);
    std::shuffle(set_labels.begin(), set_labels.end(), rng);
  }

  void ClassifierTrainingSet::print(std::ostream &os) const {
    for (size_t i = 0; i < inputs.size(); i++) {
      os << "\tinput_id: " << inputs_id[i] << ", label: " << *set_labels[i] << std::endl;
    }
  }
  void ClassifierTrainingSet::unload() {
    InputSet::unload();
    inputs_id.clear();
    set_labels.clear();
  }
}   // namespace control::classifier