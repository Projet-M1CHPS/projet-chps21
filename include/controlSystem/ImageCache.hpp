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

  class AbstractImageCache {
  public:
    enum class ScalePolicy { minimum, maximum, none };

    AbstractImageCache() : scale_policy(ScalePolicy::minimum), target_width(0), target_height(0) {}
    virtual ~AbstractImageCache() = 0;

    AbstractImageCache(AbstractImageCache const &other) = delete;
    AbstractImageCache &operator=(AbstractImageCache const &other) = delete;

    template<typename transform_iterator>
    void setupTransformEngine(transform_iterator begin, transform_iterator end);

    void setImageScaling(ScalePolicy policy) { scale_policy = policy; }

    [[nodiscard]] std::pair<size_t, size_t> getTargetSize() {
      return {target_width, target_height};
    }

    void setTargetSize(size_t width, size_t height) {
      std::tie(target_width, target_height) = {width, height};
    }

  protected:
    virtual void setupResizeTransform();

    ImageCacheStats stats;
    ScalePolicy scale_policy;
    size_t target_width, target_height;

    std::unique_ptr<image::transform::TransformEngine> engine;
  };

  class AbstractTrainingCache : public AbstractImageCache {
  public:
    AbstractTrainingCache() : is_init(false) {}
    ~AbstractTrainingCache() override = 0;

    virtual bool init() = 0;

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
    std::vector<std::pair<std::filesystem::path, size_t>> eval_set;
    std::vector<std::pair<std::filesystem::path, size_t>> training_set;

    void setupResizeTransform() override;

    bool is_init;
  };

  class TrainingStash final : public AbstractTrainingCache {
  public:
    explicit TrainingStash(std::filesystem::path const &input_path, bool shuffle_input = false);
    ~TrainingStash() override = default;

    bool init() override;

    image::GrayscaleImage const &getEval(size_t index) override;
    image::GrayscaleImage const &getTraining(size_t index) override;

  private:
    std::vector<image::GrayscaleImage> loaded_eval_set;
    std::vector<image::GrayscaleImage> loaded_training_set;
  };

}   // namespace control