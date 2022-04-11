#include "TrainingCollectionLoader.hpp"

namespace fs = std::filesystem;

namespace control {

  TrainingCollectionLoader::TrainingCollectionLoader(size_t tensor_size, size_t input_width,
                                                     size_t input_height)
      : input_set_loader(tensor_size, input_width, input_height) {}

  TrainingCollection TrainingCollectionLoader::load(const std::filesystem::path &path) {
    TrainingCollection res(input_set_loader.getInputWidth(), input_set_loader.getInputHeight());

    res.training_set = input_set_loader.load(path / "train", /*load_classes*/ true,
                                             /* shuffle samples */ true);

    res.eval_set = input_set_loader.load(path / "eval", /*load_classes*/ true,
                                         /* shuffle samples */ true);

    if (res.training_set.getClasses().size() != res.eval_set.getClasses().size())
      throw std::runtime_error("TrainingCollectionLoader::load: Training and eval sets have "
                               "different number of classes");
    if (res.training_set.getClasses().empty())
      throw std::runtime_error("TrainingCollectionLoader::load: Training and eval sets have "
                               "no classes");

    return res;
  }

}   // namespace control
