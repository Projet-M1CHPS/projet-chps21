#include "controlSystem/setLoader.hpp"

namespace fs = std::filesystem;

namespace control {


  TrainingSet control::ImageTrainingSetLoader::load(const std::filesystem::path &input_path,
                                                    bool verbose, std::ostream *out) {
    if (not fs::exists(input_path) or not fs::exists(input_path / "eval") or
        not fs::exists(input_path / "train"))
      throw std::invalid_argument(
              "ImageTrainingSetLoader: The input path does not exist or is invalid");

    TrainingSet res;
    image::transform::Resize resize_tr(target_width, target_height);
    size_t bytes;

    // List every categories in both training and eval
    std::vector<std::filesystem::path> training_categories, eval_categories;
    for (const auto &it : fs::directory_iterator(input_path / "train")) {
      if (fs::is_directory(it)) training_categories.push_back(it.path());
    }

    for (const auto &it : fs::directory_iterator(input_path / "eval")) {
      if (fs::is_directory(it)) eval_categories.push_back(it.path());
    }

    // Sort the categories by name and assign them a category index by order
    std::sort(training_categories.begin(), training_categories.end());
    std::sort(eval_categories.begin(), eval_categories.end());

    // Load the training set
    if (verbose)
      *out << "ImageTrainingSetLoader: Loading " << training_categories.size()
           << " categories from training set... (This can take a while)" << std::endl;
    for (size_t cat = 0; auto const &category : training_categories) {
      for (auto &it : fs::directory_iterator(category)) {
        if (not fs::is_regular_file(it)) continue;
        auto image = image::ImageSerializer::load(it.path());
        pre_process.apply(image);
        resize_tr.transform(image);
        post_process.apply(image);

        auto mat = image::imageToMatrix<float>(image);
        res.appendToTrainingSet(it.path(), cat, std::move(mat));
      }
      cat++;
    }

    // Load the eval set
    if (verbose)
      *out << "ImageTrainingSetLoader: Loading " << eval_categories.size()
           << " categories from eval set... (This can take a while)" << std::endl;
    for (size_t cat = 0; auto const &category : eval_categories) {
      for (auto &it : fs::directory_iterator(category)) {
        if (not fs::is_regular_file(it)) continue;
        auto image = image::ImageSerializer::load(it.path());
        pre_process.apply(image);
        resize_tr.transform(image);
        post_process.apply(image);

        auto mat = image::imageToMatrix<float>(image);
        res.appendToEvalSet(it.path(), cat, std::move(mat));
      }
      cat++;
    }

    // Set the categories names
    if (eval_categories.size() != training_categories.size() and verbose) {
      *out << "ImageTrainingSetLoader: Warning: The number of categories in training and eval set "
              "are not equal"
           << std::endl;
    }
    auto &categories = training_categories.size() > eval_categories.size() ? training_categories
                                                                           : eval_categories;
    std::vector<std::string> category_names;
    for (auto const &category : categories) {
      category_names.push_back(category.filename().string());
    }
    res.setCategories(category_names.begin(), category_names.end());

    return res;
  }
}   // namespace control
