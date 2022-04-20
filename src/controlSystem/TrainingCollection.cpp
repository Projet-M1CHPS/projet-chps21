#include "TrainingCollection.hpp"


namespace control {
  void TrainingCollection::makeTrainingTargets() {
    training_targets.clear();
    size_t nclass = training_set.getClasses().size();

    for (size_t sample_index = 0; auto &tensor : training_set.getTensors()) {
      size_t size = tensor.getDepth();
      math::clFTensor buf(nclass, 1, size);

      for (size_t j = 0; j < size; j++) {
        math::FloatMatrix mat(nclass, 1);
        mat.fill(0.0f);
        mat(training_set.getClassOf(sample_index), 0) = 1.0f;
        buf[j] = mat;
        sample_index++;
      }
      training_targets.push_back(std::move(buf));
    }
  }
}   // namespace control