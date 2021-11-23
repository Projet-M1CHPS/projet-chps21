#include "controlSystem/TrainingImageCache.h"
#include <filesystem>
#include <list>
#include <random>

using namespace image;
namespace fs = std::filesystem;

namespace control {
  namespace {

    void loadFromFolder(fs::path const &input_path,
                        std::vector<std::pair<std::filesystem::path, size_t>> &vec) {
      std::list<std::filesystem::path> categories_subfolders;
      for (const auto &dirs : fs::directory_iterator(input_path)) {
        if (dirs.is_directory()) categories_subfolders.push_front(dirs);
      }

      categories_subfolders.sort();

      for (size_t category = 0; auto const &dir : categories_subfolders) {
        for (auto const &entry : fs::directory_iterator(dir))
          if (entry.is_regular_file()) vec.emplace_back(entry, category);
        category++;
      }
    }
  }   // namespace

  TrainingImageStash::TrainingImageStash(std::filesystem::path cache_path, std::filesystem::path const &input_path,
                         bool shuffle_input)
      : TrainingImageCache(std::move(cache_path)) {
    if (not fs::exists(input_path) or not fs::exists(input_path / "eval") or
        not fs::exists(input_path / "train"))
      throw std::runtime_error("TrainingImageStash: input path doesn't exist or is invalid");

    loadFromFolder(input_path / "train", training_set);
    loadFromFolder(input_path / "eval", eval_set);

    if (shuffle_input) {
      std::random_device rd;
      std::mt19937 g(rd());

      std::shuffle(training_set.begin(), training_set.end(), g);
      std::shuffle(eval_set.begin(), eval_set.end(), g);
    }
  }

  bool TrainingImageStash::warmup() {
    loaded_eval_set.reserve(eval_set.size());
    loaded_training_set.reserve(training_set.size());

    unsigned long long cache_size = 0_byte;
    size_t i = 1, total = eval_set.size() + training_set.size();
    for (auto const &entry : eval_set) {
      loaded_eval_set.push_back(ImageSerializer::load(entry.first));
      cache_size += loaded_eval_set.back().getSize();
      std::cout << "Loading Image[" << i << "/" << total
                << "] from eval set, cache_size: " << cache_size / 1_mb << "mb\n";
      i++;
    }

    for (auto const &entry : training_set) {
      loaded_training_set.push_back(ImageSerializer::load(entry.first));
      cache_size += loaded_training_set.back().getSize();
      std::cout << "Loading Image[" << i << "/" << total
                << "] from train set, cache_size: " << cache_size / 1_mb << "mb\n";
      i++;
    }
    return true;
  }

  image::GrayscaleImage const &TrainingImageStash::getEval(size_t index) {}

  image::GrayscaleImage const &TrainingImageStash::getTraining(size_t index) {}
}   // namespace control
