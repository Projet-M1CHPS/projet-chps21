#include "TrainingCollection.hpp"


namespace control {
  void TrainingCollection::makeTrainingTargets() {
    training_targets.clear();
    size_t nclass = training_set.getClasses().size();

    std::cout << "TrainingCollection::makeTrainingTargets()" << std::endl;
    std::cout << "  nclass = " << nclass << std::endl;
    std::cout << "  ntraining = " << training_set.getTensorCount() << std::endl;
    assert(training_set.getClasses().size() == training_set.getTensorCount());

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
    int rank = 0;
    tscl::logger("[P" + std::to_string(rank) + "]: Training set {" +
                 std::to_string(training_set.getInputWidth()) + "x" +
                 std::to_string(training_set.getInputHeight()) + "}: ");
    tscl::logger("  Classes{" + std::to_string(training_set.getClasses().size()) + "}: ");
    for (auto &c : training_set.getClasses()) { tscl::logger(c + " "); }
    tscl::logger("  Tensors{" + std::to_string(training_set.getTensorCount()) + "}: ");
    for (auto &t : training_set.getTensors())
      tscl::logger("Tensor[" + std::to_string(t.getRows()) + ";" + std::to_string(t.getCols()) +
                   ";" + std::to_string(t.getDepth()) + "] {" + std::to_string(t.sum()) + "}");
    tscl::logger("  Targets{" + std::to_string(training_targets.size()) + "}: ");
    for (auto &t : training_targets)
      tscl::logger("Target[" + std::to_string(t.getRows()) + ";" + std::to_string(t.getCols()) +
                   ";" + std::to_string(t.getDepth()) + "] {" + std::to_string(t.sum()) + "}");
    tscl::logger("  Samples{" + std::to_string(training_set.getSamples().size()) + "}");
    tscl::logger("");
  }

  void TrainingCollection::split(int n, std::vector<TrainingCollection> &sub_collections) const {
    assert(sub_collections.empty());
    std::cout << "> TrainingCollection::split() Splitting training set into " << n << " parts"
              << std::endl;

    for (size_t i = 0; i < n; i++)
      sub_collections.emplace_back(
              TrainingCollection(training_set.getInputWidth(), training_set.getInputHeight()));

    std::vector<InputSet> split_training_set;
    training_set.split(n, split_training_set);

    for (size_t i = 0; i < n; i++) {
      sub_collections.at(i).training_set = std::move(split_training_set.at(i));
      sub_collections.at(i).makeTrainingTargets();
    }

    std::cout << "< TrainingCollection::split() Training split into " << n << " parts" << std::endl;
  }
}   // namespace control