#include "controlSystem/ImageCache.hpp"
#include <filesystem>
#include <list>
#include <random>

using namespace image;
namespace fs = std::filesystem;

namespace control {
  namespace {

    template<typename inserter>
    void enumerateFilesInFolder(fs::path const &input_path, inserter insert_iterator) {
      std::list<std::filesystem::path> categories_subfolders;
      for (const auto &dirs : fs::directory_iterator(input_path)) {
        if (dirs.is_directory()) categories_subfolders.push_front(dirs);
      }

      categories_subfolders.sort();

      for (size_t category = 0; auto const &dir : categories_subfolders) {
        for (auto const &entry : fs::directory_iterator(dir))
          if (entry.is_regular_file()) *insert_iterator = std::make_pair(entry, category);
        category++;
      }
    }

    template<typename insert_iterator>
    bool loadImage(fs::path const &path, insert_iterator inserter,
                   image::transform::TransformEngine *te = nullptr) {
      auto image = ImageSerializer::load(path);
      if (not image.getSize()) return false;
      if (te) te->apply(image);
      *inserter = std::move(image);
      return true;
    }

    template<typename iterator>
    std::pair<size_t, size_t> findSmallestDimensions(iterator begin, iterator end) {
      int min_w = 0, min_h = 0, buffer;

      std::tie(min_w, min_h, buffer) = ImageSerializer::loadInfo(begin->first);
      begin++;

      auto smallest = [&](std::pair<fs::path, size_t> const &pair) {
        auto [width, height, tmp] = ImageSerializer::loadInfo(pair.first);
        min_w = std::min(width, min_w);
        min_h = std::min(height, min_h);
      };

      std::for_each(begin, end, smallest);
      return {min_w, min_h};
    }

    template<typename iterator>
    std::pair<size_t, size_t> findLargestDimensions(iterator begin, iterator end) {
      int max_w = 0, max_h = 0, buffer;

      std::tie(max_w, max_h, buffer) = ImageSerializer::loadInfo(begin->first);
      begin++;

      auto smallest = [&](std::pair<fs::path, size_t> const &pair) {
        auto [width, height, tmp] = ImageSerializer::loadInfo(pair.first);
        max_w = std::max(width, max_w);
        max_h = std::max(height, max_h);
      };

      std::for_each(begin, end, smallest);
      return {max_w, max_h};
    }

  }   // namespace

  AbstractImageCache::~AbstractImageCache() {}
  AbstractTrainingCache::~AbstractTrainingCache() {}

  void AbstractImageCache::setupResizeTransform() {
    if (scale_policy == ScalePolicy::none or target_height == 0 or target_width == 0) return;

    if (not engine) engine = std::make_unique<transform::TransformEngine>();

    engine->addTransformation(std::make_shared<transform::Resize>(target_width, target_height));
  }

  void AbstractTrainingCache::setupResizeTransform() {
    size_t width = 0, height = 0;

    if (target_width or target_height) {
      if (target_height != 0 and target_width != 0) AbstractImageCache::setupResizeTransform();
      else
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

  TrainingStash::TrainingStash(std::filesystem::path const &input_path, bool shuffle_input) {
    if (not fs::exists(input_path) or not fs::exists(input_path / "eval") or
        not fs::exists(input_path / "train"))
      throw std::runtime_error("TrainingImageStash: input path doesn't exist or is invalid");

    enumerateFilesInFolder(input_path / "train", std::back_inserter(training_set));
    enumerateFilesInFolder(input_path / "eval", std::back_inserter(eval_set));

    if (shuffle_input) {
      std::random_device rd;
      std::mt19937 g(rd());

      std::shuffle(training_set.begin(), training_set.end(), g);
      std::shuffle(eval_set.begin(), eval_set.end(), g);
    }
  }

  bool TrainingStash::init() {
    if (is_init) return true;

    loaded_eval_set.reserve(eval_set.size());
    loaded_training_set.reserve(training_set.size());


    setupResizeTransform();
    auto engine_ptr = engine.get();

    size_t cache_size = 0, counter = 0, total = eval_set.size() + training_set.size();
    for (auto const &pair : eval_set) {
      counter++;
      bool res = loadImage(pair.first, std::back_inserter(loaded_eval_set), engine_ptr);
      if (not res) std::cerr << "Failed to load image " << pair.first << "." << std::endl;
      else
        cache_size += loaded_eval_set.back().getSize();
      std::cout << "[" << counter << "/" << total << "] Loading " << pair.first
                << " cache_size: " << cache_size / 1_mb << "mb\n";
    }

    for (auto const &pair : training_set) {
      counter++;
      bool res = loadImage(pair.first, std::back_inserter(loaded_training_set), engine_ptr);
      if (not res) std::cerr << "Failed to load image " << pair.first << "." << std::endl;
      else
        cache_size += loaded_training_set.back().getSize();
      std::cout << "[" << counter << "/" << total << "] Loading " << pair.first
                << " cache_size: " << cache_size / 1_mb << "mb\n";
    }

    return true;
  }

  image::GrayscaleImage const &TrainingStash::getEval(size_t index) {
    return loaded_eval_set[index];
  }

  image::GrayscaleImage const &TrainingStash::getTraining(size_t index) {
    return loaded_training_set[index];
  }
}   // namespace control
