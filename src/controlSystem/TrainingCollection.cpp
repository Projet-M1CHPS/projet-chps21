#include "TrainingCollection.hpp"


namespace control {
  void TrainingCollection::makeTrainingTargets() {
    training_targets.clear();
    size_t nclass = training_set.getClasses().size();
    assert(training_set.getClasses().size() == training_set.getTensors().size());

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

  void TrainingCollection::display() const {
    tscl::logger("Training set: ");
    tscl::logger("  Classes{" + std::to_string(training_set.getClasses().size()) + "}: ");
    for (auto &c : training_set.getClasses()) { tscl::logger(c + " "); }
    tscl::logger("  Tensors{" + std::to_string(training_set.getTensorCount()) + "}: ");
    for (auto &t : training_set.getTensors()) {
      tscl::logger("Tensor[" + std::to_string(t.getRows()) + ";" + std::to_string(t.getCols()) +
                   ";" + std::to_string(t.getDepth()) + "] {" + std::to_string(t.sum()) + "}");
    }
    tscl::logger("");
  }

  std::vector<TrainingCollection> TrainingCollection::split(int n) const {
    std::cout << "> TrainingCollection::split() Splitting training set into " << n << " parts"
              << std::endl;

    std::vector<TrainingCollection> collections;
    for (size_t i = 0; i < n; i++)
      collections.emplace_back(
              TrainingCollection(training_set.getInputWidth(), training_set.getInputHeight()));

    auto split_training_set = training_set.split(n);

    for (size_t i = 0; i < n; i++) {
      std::cout << "\t>>TrainingCollection::split() Part " << i << " of " << n - 1 << std::endl;
      collections.at(i).training_set = std::move(split_training_set.at(i));
      collections.at(i).makeTrainingTargets();
      std::cout << "\t<<TrainingCollection::split() Part " << i << " done." << std::endl;
    }

    std::cout << "< TrainingCollection::split() Training split into " << n << " parts" << std::endl;
    return collections;
  }
}   // namespace control