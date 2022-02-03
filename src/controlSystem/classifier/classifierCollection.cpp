
#include "classifierCollection.hpp"
#include "tscl.hpp"
#include <clUtils/clWrapper.hpp>
#include <functional>

namespace fs = std::filesystem;
namespace tr = image::transform;

namespace control::classifier {
  namespace {

    bool loadImageToclMatrix(math::clMatrix &res, const fs::path &path,
                             const tr::TransformEngine &preprocess,
                             const tr::TransformEngine &postprocess,
                             const tr::Transformation &resize, float normalization_factor,
                             utils::clWrapper &wrapper, cl::CommandQueue &queue) {
      try {
        // Load the image and apply the transformations pipeline
        image::GrayscaleImage img = image::ImageSerializer::load(path);
        preprocess.apply(img);
        resize.transform(img);
        postprocess.apply(img);

        // Allocate a new cl matrix
        res = math::clMatrix(img.getWidth() * img.getHeight(), 1, wrapper.getContext());
        // Convert the char array to a cl matrix
        cl::Program program = wrapper.getProgram("loadCharToFloat");
        cl::Kernel kernel(program, "loadCharToFloat");
        kernel.setArg(0, res);
        kernel.setArg(1, img.getData());

        // Normalize the matrix by a given factor
        kernel.setArg(2, normalization_factor);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                   cl::NDRange(img.getWidth() * img.getHeight()), cl::NullRange);
        return true;
      } catch (std::runtime_error &e) {
        tscl::logger("CITCLoader::loadSet: Error loading image " + path.string() + ": " + e.what(),
                     tscl::Log::Warning);
        tscl::logger("CITCLoader::loadSet: Skipping image", tscl::Log::Warning);
      }
      return false;
    }
  }   // namespace

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

    // We expect both sets to have the same classes
    // So we throw an exception if they don't match
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

  void CITCLoader::loadSet(ClassifierTrainingSet &res, const std::filesystem::path &input_path,
                           utils::clWrapper &wrapper) {
    image::transform::Resize resize(target_width, target_height);

    cl::Context context = wrapper.getContext();
    // Create an out-of-order queue
    cl::CommandQueue queue(context, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

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

      // We iterate on the directory, creating an OpenCL buffer for each image
      for (auto &entry : fs::directory_iterator(target_path)) {
        if (not entry.is_regular_file()) continue;

        math::clMatrix tmp;
        if (loadImageToclMatrix(tmp, entry.path(), pre_process, post_process, resize, 255, wrapper,
                                queue))
          res.append(0, &c.second, tmp);
      }
    }
    // Wait for all jobs to finish
    queue.finish();
  }

  std::unique_ptr<CTCollection> CITCLoader::load(const std::filesystem::path &input_path,
                                                 utils::clWrapper &wrapper) {
    if (not fs::exists(input_path) or not fs::exists(input_path / "eval") or
        not fs::exists(input_path / "train")) {
      tscl::logger("CITCLoader: " + input_path.string() + " is not a valid CITC directory",
                   tscl::Log::Error);
      throw std::invalid_argument(
              "ImageTrainingSetLoader: The input path does not exist or is invalid");
    }

    tscl::logger("Loading Classifier training collection from " + input_path.string(),
                 tscl::Log::Debug);
    // If the user didn't specify classes, we extract them from the inpuy directory layout
    // Since each class must have one folder
    if (not classes) {
      tscl::logger("Looking for classes", tscl::Log::Debug);
      loadClasses(input_path);
      tscl::logger("Found " + std::to_string(classes->size()) + " classes", tscl::Log::Trace);
    }

    if (classes->empty()) {
      tscl::logger("CITCLoader: Need at-least one class, none were found", tscl::Log::Error);
      throw std::runtime_error("CITCLoader: Need at-least one class, none were found");
    }

    auto res = std::make_unique<CTCollection>(classes);

    // Load both sets
    loadSet(res->getEvalSet(), input_path, wrapper);
    loadSet(res->getTrainingSet(), input_path, wrapper);

    tscl::logger("Loaded " + std::to_string(res->getTrainingSet().size()) + " inputs",
                 tscl::Log::Trace);

    return res;
  }

}   // namespace control::classifier