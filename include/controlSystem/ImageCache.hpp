#pragma once

#include "Image.hpp"
#include "Transform.hpp"
#include <filesystem>
#include <iostream>
#include <list>
#include <random>
#include <vector>

namespace control {

  inline unsigned long long operator"" _gb(unsigned long long x) { return x * 1000000000; }
  inline unsigned long long operator"" _mb(unsigned long long x) { return x * 1000000; }
  inline unsigned long long operator"" _kb(unsigned long long x) { return x * 1000; }
  inline unsigned long long operator"" _byte(unsigned long long x) { return x; }

  namespace {

    template<typename inserter>
    void enumerateFilesInFolder(std::filesystem::path const &input_path, inserter insert_iterator) {
      std::list<std::filesystem::path> categories_subfolders;
      for (const auto &dirs : std::filesystem::directory_iterator(input_path)) {
        if (dirs.is_directory()) categories_subfolders.push_front(dirs);
      }

      categories_subfolders.sort();

      for (size_t category = 0; auto const &dir : categories_subfolders) {
        for (auto const &entry : std::filesystem::directory_iterator(dir))
          if (entry.is_regular_file()) *insert_iterator = std::make_pair(entry, category);
        category++;
      }
    }

    template<typename real, typename insert_iterator>
    bool loadImage(std::filesystem::path const &path, insert_iterator inserter,
                   image::transform::TransformEngine *te = nullptr) {
      auto image = image::ImageSerializer::load(path);
      if (not image.getSize()) return false;
      if (te) te->apply(image);
      *inserter = std::move(image::imageToMatrix<real>(image, 255));
      return true;
    }

    template<typename iterator>
    std::pair<size_t, size_t> findSmallestDimensions(iterator begin, iterator end) {
      int min_w = 0, min_h = 0, buffer;

      std::tie(min_w, min_h, buffer) = image::ImageSerializer::loadInfo(begin->first);
      begin++;

      auto smallest = [&](std::pair<std::filesystem::path, size_t> const &pair) {
        auto [width, height, tmp] = image::ImageSerializer::loadInfo(pair.first);
        min_w = std::min(width, min_w);
        min_h = std::min(height, min_h);
      };

      std::for_each(begin, end, smallest);
      return {min_w, min_h};
    }

    template<typename iterator>
    std::pair<size_t, size_t> findLargestDimensions(iterator begin, iterator end) {
      int max_w = 0, max_h = 0, buffer;

      std::tie(max_w, max_h, buffer) = image::ImageSerializer::loadInfo(begin->first);
      begin++;

      auto smallest = [&](std::pair<std::filesystem::path, size_t> const &pair) {
        auto [width, height, tmp] = image::ImageSerializer::loadInfo(pair.first);
        max_w = std::max(width, max_w);
        max_h = std::max(height, max_h);
      };

      std::for_each(begin, end, smallest);
      return {max_w, max_h};
    }

  }   // namespace

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

    void setImageScaling(ScalePolicy policy) {
      scale_policy = policy;
      target_width = target_height = 0;
    }

    [[nodiscard]] std::pair<size_t, size_t> getTargetSize() {
      return {target_width, target_height};
    }

    void setTargetSize(size_t width, size_t height) {
      scale_policy = ScalePolicy::none;
      std::tie(target_width, target_height) = {width, height};
    }

  protected:
    virtual void setupResizeTransform() {
      if (not engine) engine = std::make_unique<image::transform::TransformEngine>();

      engine->addTransformation(
              std::make_shared<image::transform::Resize>(target_width, target_height));
    }

    ImageCacheStats stats;
    ScalePolicy scale_policy;
    size_t target_width, target_height;

    std::unique_ptr<image::transform::TransformEngine> engine;
  };

  template<typename real>
  class AbstractTrainingCache : public AbstractImageCache {
  public:
    AbstractTrainingCache() : is_init(false) {}
    ~AbstractTrainingCache() override = default;

    virtual bool init() = 0;

    virtual math::Matrix<real> const &getEval(size_t index) = 0;
    [[nodiscard]] size_t getEvalType(size_t index) const {
      if (index > eval_set.size())
        throw std::out_of_range("TrainingImageCache: Invalid index for eval set types");

      return eval_set[index].second;
    }

    virtual math::Matrix<real> const &getTraining(size_t index) = 0;
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

    void setupResizeTransform() override {
      size_t width = 0, height = 0;

      if (scale_policy == ScalePolicy::none) {
        if (target_height != 0 and target_width != 0) {
          AbstractImageCache::setupResizeTransform();
          return;
        } else
          throw std::runtime_error(
                  "setupResizeTransform(): One of the target dimension is incorrect");
      }

      if (scale_policy == ScalePolicy::minimum) {
        auto min = findSmallestDimensions(training_set.begin(), training_set.end());
        std::tie(target_width, target_height) = min;
      } else if (scale_policy == ScalePolicy::maximum) {
        auto max = findLargestDimensions(training_set.begin(), training_set.end());
        std::tie(target_width, target_height) = max;
      } else
        return;
      AbstractImageCache::setupResizeTransform();
    }

    bool is_init;
  };

  template<typename real>
  class TrainingStash final : public AbstractTrainingCache<real> {
    using path = std::filesystem::path;

  public:
    explicit TrainingStash(path const &input_path, bool shuffle_input = false) {
      if (not std::filesystem::exists(input_path) or
          not std::filesystem::exists(input_path / "eval") or
          not std::filesystem::exists(input_path / "train"))
        throw std::runtime_error("TrainingImageStash: input path doesn't exist or is invalid");

      enumerateFilesInFolder(input_path / "train", std::back_inserter(this->training_set));
      enumerateFilesInFolder(input_path / "eval", std::back_inserter(this->eval_set));

      if (shuffle_input) {
        std::random_device rd;
        std::mt19937 g(rd());

        std::shuffle(this->training_set.begin(), this->training_set.end(), g);
        std::shuffle(this->eval_set.begin(), this->eval_set.end(), g);
      }
    }

    ~TrainingStash() override = default;

    bool init() override {
      if (this->is_init) return true;

      loaded_eval_set.reserve(this->eval_set.size());
      loaded_training_set.reserve(this->training_set.size());


      this->setupResizeTransform();
      auto engine_ptr = this->engine.get();

      size_t cache_size = 0, counter = 0, total = this->eval_set.size() + this->training_set.size();
      for (auto const &pair : this->eval_set) {
        counter++;
        bool res = loadImage<real>(pair.first, std::back_inserter(loaded_eval_set), engine_ptr);
        if (not res) std::cerr << "Failed to load image " << pair.first << "." << std::endl;
        else
          cache_size += loaded_eval_set.back().getSize() * sizeof(real);
        std::cout << "[" << counter << "/" << total << "] Loading " << pair.first
                  << " cache_size: " << cache_size / 1_mb << "mb\n";
      }

      for (auto const &pair : this->training_set) {
        counter++;
        bool res = loadImage<real>(pair.first, std::back_inserter(loaded_training_set), engine_ptr);
        if (not res) std::cerr << "Failed to load image " << pair.first << "." << std::endl;
        else
          cache_size += loaded_training_set.back().getSize();
        std::cout << "[" << counter << "/" << total << "] Loading " << pair.first
                  << " cache_size: " << cache_size / 1_mb << "mb\n";
      }

      return true;
    }

    math::Matrix<real> const &getEval(size_t index) override {
      if (index > loaded_eval_set.size()) throw std::out_of_range("Invalid image index");

      return loaded_eval_set[index];
    }

    math::Matrix<real> const &getTraining(size_t index) override {
      if (index > loaded_training_set.size()) throw std::out_of_range("Invalid image index");

      return loaded_training_set[index];
    }

  private:
    std::vector<math::Matrix<real>> loaded_eval_set;
    std::vector<math::Matrix<real>> loaded_training_set;
  };

}   // namespace control