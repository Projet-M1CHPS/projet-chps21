#pragma once

#include "Image.hpp"
#include "Transform.hpp"
#include <filesystem>
#include <iostream>
#include <vector>

namespace control {

  inline unsigned long long operator"" _gb(unsigned long long x) { return x * 1000000000; }
  inline unsigned long long operator"" _mb(unsigned long long x) { return x * 1000000; }
  inline unsigned long long operator"" _kb(unsigned long long x) { return x * 1000; }
  inline unsigned long long operator"" _byte(unsigned long long x) { return x; }


  class ImageCacheStats {};

  class TrainingImageCache {
  public:
    explicit TrainingImageCache(std::filesystem::path cache_path)
        : cache_path(std::move(cache_path)) {}

    TrainingImageCache(TrainingImageCache const &other) = delete;
    TrainingImageCache &operator=(TrainingImageCache const &other) = delete;

    TrainingImageCache(TrainingImageCache &&other) noexcept = default;
    TrainingImageCache &operator=(TrainingImageCache &&other) noexcept = default;

    virtual bool warmup() = 0;

    virtual image::GrayscaleImage const &getEval(size_t index) = 0;
    [[nodiscard]] size_t getEvalType(size_t index) const {
      if (index > eval_set.size())
        throw std::out_of_range("TrainingImageCache: Invalid index for eval set types");

      return eval_set[index].second;
    }

    virtual image::GrayscaleImage const &getTraining(size_t index) = 0;
    [[nodiscard]] size_t getTrainingType(size_t index) const {
      if (index > training_set.size())
        throw std::out_of_range("TrainingImageCache: Invalid index for training set types");

      return training_set[index].second;
    }

    [[nodiscard]] size_t getEvalSetSize() { return eval_set.size(); }

    [[nodiscard]] size_t getTrainingSetSize() { return training_set.size(); }

    [[nodiscard]] ImageCacheStats const &getStats() { return stats; }

  protected:
    std::shared_ptr<image::transform::TransformEngine> engine;

    std::vector<std::pair<std::filesystem::path, size_t>> eval_set;
    std::vector<std::pair<std::filesystem::path, size_t>> training_set;

    std::filesystem::path cache_path;
    ImageCacheStats stats;
  };

  class TrainingImageStash final : public TrainingImageCache {
  public:
    explicit TrainingImageStash(std::filesystem::path cache_path,
                                std::filesystem::path const &input_path,
                                bool shuffle_input = false);

    bool warmup() override;

    image::GrayscaleImage const &getEval(size_t index) override;
    image::GrayscaleImage const &getTraining(size_t index) override;

  private:
    std::vector<image::GrayscaleImage> loaded_eval_set;
    std::vector<image::GrayscaleImage> loaded_training_set;
  };

}   // namespace control