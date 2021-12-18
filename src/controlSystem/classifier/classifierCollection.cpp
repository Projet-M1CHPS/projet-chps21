
#include "classifierCollection.hpp"
#include "tscl.hpp"

namespace fs = std::filesystem;

namespace control::classifier {

  CTCollection::CTCollection(std::shared_ptr<ClassifierClassLabelList> classes)
      : class_list(std::move(classes)) {}

  std::ostream &operator<<(std::ostream &os, CTCollection const &set) {
    os << "Classifier training set: " << std::endl;
    os << "\tTraining set contains " << set.training_set.size() << " elements" << std::endl;
    os << "\tEvaluation set contains " << set.eval_set.size() << " elements" << std::endl;

    os << "Classes: " << std::endl;
    os << set.class_list << std::endl;

    return os;
  }


  void CITCLoader::loadClasses(fs::path const &input_path) {
    if (not fs::exists(input_path / "train")) {
      tscl::logger("Train folder not found in " + input_path.string(), tscl::Log::Error);
      throw std::runtime_error("CITSLoader: train folder not found");
    }

    if (not fs::exists(input_path / "eval")) {
      tscl::logger("Eval folder not found in " + input_path.string(), tscl::Log::Error);
      throw std::runtime_error("CITSLoader: eval folder not found");
    }

    std::vector<std::filesystem::path> training_classes, eval_classes;
    std::for_each(fs::directory_iterator(input_path / "train"), fs::directory_iterator(),
                  [&training_classes](const fs::path &p) {
                    if (fs::is_directory(p)) training_classes.push_back(p.filename());
                  });

    std::for_each(fs::directory_iterator(input_path / "train"), fs::directory_iterator(),
                  [&eval_classes](const fs::path &p) {
                    if (fs::is_directory(p)) eval_classes.push_back(p.filename());
                  });

    std::sort(training_classes.begin(), training_classes.end());
    std::sort(eval_classes.begin(), eval_classes.end());


    if (not std::equal(training_classes.begin(), training_classes.end(), eval_classes.begin())) {
      tscl::logger("Training and evaluation classes are not the same", tscl::Log::Error);
      size_t i = 0, j = 0;
      while (i < training_classes.size() and j < eval_classes.size()) {
        while (training_classes[i] != eval_classes[j] and i < eval_classes.size()) {
          tscl::logger("Training class " + training_classes[i].string() + " not found in eval",
                       tscl::Log::Error);
          i++;
        }
        i++;
        j++;
      }

      for (; i < training_classes.size(); i++) {
        tscl::logger("Training class " + eval_classes[i].string() + " not found in eval",
                     tscl::Log::Error);
      }
      for (; j < eval_classes.size(); j++) {
        tscl::logger("Eval class " + eval_classes[j].string() + " not found in training",
                     tscl::Log::Error);
      }
      throw std::runtime_error("CITSLoader: train and eval classes are not the same");
    }

    classes = std::make_shared<std::vector<ClassLabel>>();
    std::for_each(training_classes.begin(), training_classes.end(),
                  [this](const std::filesystem::path &p) {
                    this->classes->emplace_back(classes->size(), p.string());
                  });
  }

  void CITCLoader::loadEvalSet(CTCollection &res, const std::filesystem::path &input_path) {
    tscl::logger("Loading eval set (Hold on, this may take a while)", tscl::Log::Information);
    auto &eval_set = res.getEvalSet();
    loadSet(eval_set, input_path / "eval");
  }

  void CITCLoader::loadTrainingSet(CTCollection &res, fs::path const &input_path) {
    tscl::logger("Loading training set (Hold on, this may take a while)", tscl::Log::Information);
    auto &training_set = res.getTrainingSet();
    loadSet(training_set, input_path / "train");
  }

  void CITCLoader::loadSet(ClassifierTrainingSet &res, const std::filesystem::path &input_path) {
    image::transform::Resize resize(target_width, target_height);

    for (auto &c : *classes) {
      fs::path target_path = input_path / c.getName();
      if (not fs::exists(target_path)) {
        tscl::logger("Class " + c.getName() + " not found in " + input_path.string(),
                     tscl::Log::Error);
        throw std::runtime_error("CITSLoader: " + input_path.string() + " is missing class id: " +
                                 std::to_string(c.getId()) + " (\"" + c.getName() + "\")");
      }

      fs::create_directories("tmp_cache/" + c.getName());
      for (auto &entry : fs::directory_iterator(target_path)) {
        if (fs::is_regular_file(entry)) {
          image::GrayscaleImage img = image::ImageSerializer::load(entry);
          pre_process.apply(img);
          resize.transform(img);
          post_process.apply(img);

          image::ImageSerializer::save(
                  "tmp_cache/" + c.getName() + "/" + entry.path().filename().string(), img);

          auto mat = image::imageToMatrix<float>(img, 255);
          res.append(entry, &c, std::move(mat));
        }
      }
    }
  }

  std::shared_ptr<CTCollection> CITCLoader::load(const std::filesystem::path &input_path) {
    if (not fs::exists(input_path) or not fs::exists(input_path / "eval") or
        not fs::exists(input_path / "train")) {
      tscl::logger("CITCLoader: " + input_path.string() + " is not a valid CITC directory",
                   tscl::Log::Error);
      throw std::invalid_argument(
              "ImageTrainingSetLoader: The input path does not exist or is invalid");
    }


    tscl::logger("Loading Classifier training collection at " + input_path.string(),
                 tscl::Log::Information);
    if (not classes) {
      tscl::logger("Loading classes", tscl::Log::Debug);
      loadClasses(input_path);
      if (classes->empty()) {
        tscl::logger("CITCLoader: No classes found in " + input_path.string(), tscl::Log::Error);
        throw std::runtime_error("ImageTrainingSetLoader: No classes found");
      }
      tscl::logger("Found " + std::to_string(classes->size()) + " classes", tscl::Log::Information);
      std::stringstream ss;

      for (auto &c : *classes) ss << c.getName() << ", ";
      tscl::logger("\tClasses: " + ss.str(), tscl::Log::Debug);
    } else if (classes->empty()) {
      tscl::logger("CITCLoader: Need at-least one class, none were given", tscl::Log::Error);
      throw std::runtime_error("CITCLoader: Need at-least one class, none were given");
    }

    auto res = std::make_shared<CTCollection>(classes);

    loadEvalSet(*res, input_path);
    loadTrainingSet(*res, input_path);

    tscl::logger("Loaded " + std::to_string(res->getTrainingSet().size()) + " inputs",
                 tscl::Log::Information);

    return res;
  }

}   // namespace control::classifier