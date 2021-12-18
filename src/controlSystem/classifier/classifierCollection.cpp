
#include "classifierCollection.hpp"
#include "tscl.hpp"
#include <functional>

namespace fs = std::filesystem;

namespace control::classifier {

  CTCollection::CTCollection(std::shared_ptr<CClassLabelSet> classes)
      : class_list(std::move(classes)) {}

  std::ostream &operator<<(std::ostream &os, CTCollection const &set) {
    os << "Classifier training set: " << std::endl;
    os << "\tTraining set contains " << set.training_set.size() << " elements" << std::endl;
    os << "\tEvaluation set contains " << set.eval_set.size() << " elements" << std::endl;

    os << "Classes: " << std::endl;
    os << *set.class_list << std::endl;

    return os;
  }

  // If the user didn't specify which classes to load, we load all of them
  // by associating each class with a directory
  void CITCLoader::loadClasses(fs::path const &input_path) {
    std::vector<std::filesystem::path> training_classes, eval_classes;

    for (auto const &p : fs::directory_iterator(input_path / "eval")) {
      if (p.is_directory()) training_classes.push_back(p.path().filename());
    }

    for (auto const &p : fs::directory_iterator(input_path / "train")) {
      if (p.is_directory()) eval_classes.push_back(p.path().filename());
    }

    std::sort(training_classes.begin(), training_classes.end());
    std::sort(eval_classes.begin(), eval_classes.end());

    /** We expect both sets to have the same classes
     * So we throw an exception if they don't match
     */
    if (not std::equal(training_classes.begin(), training_classes.end(), eval_classes.begin())) {
      tscl::logger("Training and evaluation classes are not the same", tscl::Log::Error);
      throw std::runtime_error("CITSLoader: train and eval classes are not the same");
    }

    classes = std::make_shared<CClassLabelSet>();
    for (auto const &p : training_classes) {
      ClassLabel tmp(classes->size(), p.filename().string());
      classes->append(tmp);
    }
  }

  void CITCLoader::loadEvalSet(CTCollection &res, const std::filesystem::path &input_path) {
    tscl::logger("Loading eval set (Hold on, this may take a while)", tscl::Log::Debug);
    auto &eval_set = res.getEvalSet();
    loadSet(eval_set, input_path / "eval");
  }

  void CITCLoader::loadTrainingSet(CTCollection &res, fs::path const &input_path) {
    tscl::logger("Loading training set (Hold on, this may take a while)", tscl::Log::Debug);
    auto &training_set = res.getTrainingSet();
    loadSet(training_set, input_path / "train");
  }

  void CITCLoader::loadSet(ClassifierTrainingSet &res, const std::filesystem::path &input_path) {
    image::transform::Resize resize(target_width, target_height);

    // Each class is a directory
    // inside the input path
    for (auto &c : *classes) {
      auto c_name = c.second.getName();
      fs::path target_path = input_path / c_name;

      // If for any reason the class directory doesn't exist, throw an error
      if (not fs::exists(target_path)) {
        tscl::logger("Class " + c_name + " not found in " + input_path.string(), tscl::Log::Error);
        throw std::runtime_error("CITSLoader: " + input_path.string() + " is missing class id: " +
                                 std::to_string(c.first) + " (\"" + c_name + "\")");
      }

      fs::create_directories("tmp_cache/" + c_name);
      // We iterate over the content of the current class dir, applying the transformations
      // sequentially to each image
      for (auto &entry : fs::directory_iterator(target_path)) {
        if (fs::is_regular_file(entry)) {
          // Load an image as a grayscale image and apply the transformations
          image::GrayscaleImage img = image::ImageSerializer::load(entry);
          pre_process.apply(img);
          resize.transform(img);
          post_process.apply(img);

          // We then convert the image to a matrix, normalize it and add it to the training set
          // We perform normalization in the range [0, 1] so that if the network is trained with
          // a linear activation function, the gradient will not explode
          // This also works for sigmoid which will output values in the range [0, 1] anyway
          auto mat = image::imageToMatrix<float>(img, 255);

          // Move the matrix to the input set to avoid unnecessary copies
          res.append(0, &c.second, std::move(mat));
        }
      }
    }
  }

  std::unique_ptr<CTCollection> CITCLoader::load(const std::filesystem::path &input_path) {
    if (not fs::exists(input_path) or not fs::exists(input_path / "eval") or
        not fs::exists(input_path / "train")) {
      tscl::logger("CITCLoader: " + input_path.string() + " is not a valid CITC directory",
                   tscl::Log::Error);
      throw std::invalid_argument(
              "ImageTrainingSetLoader: The input path does not exist or is invalid");
    }

    tscl::logger("Loading Classifier training collection at " + input_path.string(),
                 tscl::Log::Debug);
    if (not classes) {
      tscl::logger("Loading classes", tscl::Log::Debug);
      loadClasses(input_path);
      if (classes->empty()) {
        tscl::logger("CITCLoader: No classes found in " + input_path.string(), tscl::Log::Error);
        throw std::runtime_error("ImageTrainingSetLoader: No classes found !");
      }

      tscl::logger("Found " + std::to_string(classes->size()) + " classes", tscl::Log::Debug);

    } else if (classes->empty()) {
      tscl::logger("CITCLoader: Need at-least one class, none were given", tscl::Log::Error);
      throw std::runtime_error("CITCLoader: Need at-least one class, none were given");
    }

    auto res = std::make_unique<CTCollection>(classes);

    loadEvalSet(*res, input_path);
    loadTrainingSet(*res, input_path);

    tscl::logger("Loaded " + std::to_string(res->getTrainingSet().size()) + " inputs",
                 tscl::Log::Debug);

    return res;
  }

}   // namespace control::classifier