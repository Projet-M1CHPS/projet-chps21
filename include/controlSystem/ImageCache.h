#pragma once

#include "Image.hpp"
#include <filesystem>
#include <iostream>
#include <vector>

namespace control {

  inline unsigned long long operator"" _gb(unsigned long long x) { return x * 1000000000; }
  inline unsigned long long operator"" _mb(unsigned long long x) { return x * 1000000; }
  inline unsigned long long operator"" _kb(unsigned long long x) { return x * 1000; }
  inline unsigned long long operator"" _byte(unsigned long long x) { return x; }


  class ImageCacheStats {

  };

  class ImageCache {
  public:

    ImageCache(std::filesystem::path input_path, bool shuffle_input = false);

    ImageCache(ImageCache const &other) = delete;
    ImageCache &operator=(ImageCache const &other) = delete;

    ImageCache(ImageCache &&other) noexcept;
    ImageCache &operator=(ImageCache &&other) noexcept;

    virtual bool warmup() = 0;

    virtual image::GrayscaleImage const &getEval(size_t index) = 0;
    virtual image::GrayscaleImage const &getTraining(size_t index) = 0;

    [[nodiscard]] size_t getEvalSetSize() {
      return eval_set.size();
    }

    [[nodiscard]] size_t getTrainingSetSize() {
      return training_set.size();
    }

    [[nodiscard]] ImageCacheStats const& getStats() {
      return stats;
    }

  protected:
    std::vector<std::filesystem::path> eval_set;
    std::vector<std::filesystem::path> training_set;

    std::filesystem::path cache_path;
    ImageCacheStats stats;
  };

}   // namespace control