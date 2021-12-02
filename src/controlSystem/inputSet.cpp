#include <controlSystem/inputSet.hpp>

namespace control {

  void InputSet::append(std::filesystem::path path, math::Matrix<float> &&mat) {
    inputs_path.push_back(std::move(path));
    inputs.push_back(std::move(mat));
  }

  void TrainingSet::appendToTrainingSet(std::filesystem::path path, size_t category,
                                        math::Matrix<float> &&mat) {
    training_set_files.push_back(std::move(path));
    training_set.push_back(std::move(mat));
    training_set_categories.push_back(category);
  }

  void TrainingSet::appendToEvalSet(std::filesystem::path path, size_t category,
                                    math::Matrix<float> &&mat) {
    eval_set_files.push_back(std::move(path));
    eval_set.push_back(std::move(mat));
    eval_set_categories.push_back(category);
  }

  void TrainingSet::shuffleTrainingSet(size_t seed) {
    std::mt19937_64 rng(seed);
    std::shuffle(training_set_files.begin(), training_set_files.end(), rng);
    rng.seed(seed);
    std::shuffle(training_set.begin(), training_set.end(), rng);
    rng.seed(seed);
    std::shuffle(training_set_categories.begin(), training_set_categories.end(), rng);
  }

  void TrainingSet::shuffleEvalSet(size_t seed) {
    std::mt19937_64 rng(seed);
    std::shuffle(eval_set_files.begin(), eval_set_files.end(), rng);
    rng.seed(seed);
    std::shuffle(eval_set.begin(), eval_set.end(), rng);
    rng.seed(seed);
    std::shuffle(eval_set_categories.begin(), eval_set_categories.end(), rng);
  }

  void TrainingSet::shuffleSets(size_t seed) {
    shuffleTrainingSet(seed);
    shuffleEvalSet(seed);
  }

  size_t TrainingSet::trainingSetSize() const { return training_set.size(); }

  size_t TrainingSet::evalSetSize() const { return eval_set.size(); }

  void TrainingSet::unload() { *this = std::move(TrainingSet()); }
}   // namespace control
