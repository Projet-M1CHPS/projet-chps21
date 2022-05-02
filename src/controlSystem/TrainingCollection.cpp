#include "TrainingCollection.hpp"


namespace control {
  void TrainingCollection::makeTrainingTargets() {
    tscl::logger("TrainingCollection::makeTrainingTargets()");
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

  std::vector<TrainingCollection> TrainingCollection::splitTrainingSet(size_t npart) const {
    std::vector<TrainingCollection> res;
    res.reserve(npart);

    const auto &dataset = training_set;

    size_t tensor_per_part = dataset.getTensorCount() / npart;
    size_t tensor_remainder = dataset.getTensorCount() % npart;

    size_t sample_index = 0;
    size_t tensor_index = 0;
    for (size_t i = 0; i < npart; ++i) {
      TrainingCollection collection(dataset.getInputWidth(), dataset.getInputHeight());
      collection.updateClasses(dataset.getClasses());
      size_t local_tensor_count = tensor_per_part;

      if (tensor_remainder) {
        local_tensor_count++;
        tensor_remainder--;
      }

      for (size_t j = 0; j < local_tensor_count; j++) {
        auto &tensor = dataset.getTensor(tensor_index);
        std::vector<size_t> ids;
        std::vector<long> class_ids;

        for (size_t k = 0; k < tensor.getDepth(); k++) {
          ids.push_back(dataset.getSampleId(sample_index));
          class_ids.push_back(dataset.getClassOf(sample_index));
          sample_index++;
        }
        tensor_index++;
        collection.getTrainingSet().append(tensor.shallowCopy(), ids, class_ids);
      }

      res.emplace_back(std::move(collection));
    }

    return res;
  }

}   // namespace control