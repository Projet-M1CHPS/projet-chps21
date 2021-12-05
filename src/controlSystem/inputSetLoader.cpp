#include "controlSystem/inputSetLoader.hpp"
#include <algorithm>
#include <list>

namespace fs = std::filesystem;

namespace control {


  void CITCLoader::loadClasses(fs::path const &input_path) {
    if (not fs::exists(input_path / "train"))
      throw std::runtime_error("CITSLoader: train folder not found");

    if (not fs::exists(input_path / "eval"))
      throw std::runtime_error("CITSLoader: eval folder not found");

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
      std::cerr << "CITSLoader: train and eval classes are not the same" << std::endl;
      size_t i = 0, j = 0;
      while (i < training_classes.size() and j < eval_classes.size()) {
        while (training_classes[i] != eval_classes[j] and i < eval_classes.size()) {
          std::cerr << "Class " << training_classes[i] << " is not in eval" << std::endl;
          i++;
        }
        i++;
        j++;
      }

      for (; i < training_classes.size(); i++) {
        std::cerr << "Class " << eval_classes[i] << " is not in train" << std::endl;
      }
      for (; j < eval_classes.size(); j++) {
        std::cerr << "Class " << eval_classes[i] << " is not in train" << std::endl;
      }
      throw std::runtime_error("CITSLoader: train and eval classes are not the same");
    }
    classes = std::make_shared<std::set<ClassLabel>>();

    std::for_each(training_classes.begin(), training_classes.end(),
                  [this](const std::filesystem::path &p) {
                    this->classes->insert(ClassLabel(classes->size(), p.string()));
                  });
  }

  void CITCLoader::loadEvalSet(ClassifierTrainingCollection &res,
                               const std::filesystem::path &input_path, bool verbose,
                               std::ostream *out) {
    if (verbose) *out << "\tLoading eval set... (Hold on, This may take a while)" << std::endl;
    auto &eval_set = res.getEvalSet();
    loadSet(eval_set, input_path / "eval");
  }

  void CITCLoader::loadTrainingSet(ClassifierTrainingCollection &res, fs::path const &input_path,
                                   bool verbose, std::ostream *out) {
    if (verbose) *out << "\tLoading training set... (Hold on, This may take a while)" << std::endl;
    auto &training_set = res.getTrainingSet();
    loadSet(training_set, input_path / "eval");
  }

  void CITCLoader::loadSet(ClassifierInputSet &res, const std::filesystem::path &input_path) {
    image::transform::Resize resize(target_width, target_height);

    for (auto &c : *classes) {
      fs::path target_path = input_path / c.getName();
      if (not fs::exists(target_path))
        throw std::runtime_error("CITSLoader: " + input_path.string() + " is missing class id: " +
                                 std::to_string(c.getId()) + " (\"" + c.getName() + "\")");

      for (auto &entry : fs::directory_iterator(target_path)) {
        if (fs::is_regular_file(entry)) {
          image::GrayscaleImage img = image::ImageSerializer::load(entry);
          pre_process.apply(img);
          resize.transform(img);
          post_process.apply(img);
          auto mat = image::imageToMatrix<float>(img, 255);
          res.append(entry, &c, std::move(mat));
        }
      }
    }
  }

  std::shared_ptr<ClassifierTrainingCollection>
  control::CITCLoader::load(const std::filesystem::path &input_path, bool verbose,
                            std::ostream *out) {
    if (not fs::exists(input_path) or not fs::exists(input_path / "eval") or
        not fs::exists(input_path / "train"))
      throw std::invalid_argument(
              "ImageTrainingSetLoader: The input path does not exist or is invalid");


    if (verbose) *out << "Loading Classifier training set at " << input_path << std::endl;
    if (not classes) {
      if (verbose) std::cout << "\tLoading classes..." << std::endl;
      loadClasses(input_path);
      if (classes->empty()) throw std::runtime_error("ImageTrainingSetLoader: No classes found");
      else if (verbose) {
        *out << "\tFound " << classes->size() << " classes: " << std::endl;
        for (auto &c : *classes) *out << "\t" << c << std::endl;
      }
    } else if (classes->empty())
      throw std::runtime_error("ImageTrainingSetLoader: Need at-least one class, none were given");

    auto res = std::make_shared<ClassifierTrainingCollection>(classes);

    loadEvalSet(*res, input_path, verbose, out);
    loadTrainingSet(*res, input_path, verbose, out);

    if (verbose) *out << "Done loading " << res->size() << " elements" << std::endl;

    return res;
  }
}   // namespace control
