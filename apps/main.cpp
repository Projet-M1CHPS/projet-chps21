#include "EvalController.hpp"
#include "NeuralNetwork.hpp"
#include "ProjectVersion.hpp"
#include "TrainingController.hpp"
#include "controlSystem/TrainingCollection.hpp"
#include "controlSystem/TrainingCollectionLoader.hpp"
#include "openclUtils/clPlatformSelector.hpp"
#include "tscl.hpp"

#include <iomanip>
#include <iostream>
#include <vector>

#include <CL/opencl.hpp>

using namespace control;
using namespace tscl;

void setupLogger() {
  auto &handler = logger.addHandler<StreamLogHandler>("term", std::cout);
  handler.tsType(tscl::timestamp_t::Partial);
  handler.minLvl(Log::Trace);

  auto &thandler = logger.addHandler<StreamLogHandler>("file", "logs.txt");
  thandler.minLvl(Log::Information);
}

bool createAndTrain(std::filesystem::path const &input_path,
                    std::filesystem::path const &output_path) {
  tscl::logger("Current version: " + tscl::Version::current.to_string(), tscl::Log::Debug);
  tscl::logger("Fetching input from  " + input_path.string(), tscl::Log::Debug);
  tscl::logger("Output path: " + output_path.string(), tscl::Log::Debug);

  if (not std::filesystem::exists(output_path)) std::filesystem::create_directories(output_path);

  constexpr int kImageSize = 64;
  // Ensure this is the same size as the batch size
  constexpr int kTensorSize = 8;

  tscl::logger("Loading dataset", tscl::Log::Debug);
  TrainingCollectionLoader loader(kTensorSize, kImageSize, kImageSize);
  auto &pre_engine = loader.getPreProcessEngine();
  // Add preprocessing transformations here
  pre_engine.addTransformation(std::make_shared<image::transform::Inversion>());

  auto &engine = loader.getPostProcessEngine();
  // Add postprocessing transformations here
  engine.addTransformation(std::make_shared<image::transform::BinaryScale>());

  TrainingCollection training_collection = loader.load(input_path);

  tscl::logger("Training set size: " +
                       std::to_string(training_collection.getTrainingSet().getSize()),
               tscl::Log::Trace);
  tscl::logger("Testing set size: " +
                       std::to_string(training_collection.getEvaluationSet().getSize()),
               tscl::Log::Trace);


  // Create a correctly-sized topology
  nnet::MLPTopology topology = {kImageSize * kImageSize, 256, 64};
  topology.pushBack(training_collection.getClassCount());

  // auto model = nnet::MLPModel::randomReluSigmoid(topology);
  auto model = nnet::MLPModel::random(topology, af::ActivationFunctionType::leakyRelu);
  // auto model = std::make_unique<nnet::MLPModel>();
  // model->load("michal.nnet");

  auto optimizer = nnet::MLPOptimizer::make<nnet::SGDOptimization>(*model, 0.08);
  // ParallelScheduler scheduler(batch, input, target);

  tscl::logger("Creating controller", tscl::Log::Trace);
  // EvalController controller(output_path, model.get(), &training_collection.getEvaluationSet());
  TrainingController controller(output_path, *model, *optimizer, training_collection, 1000);
  ControllerResult res = controller.run();

  if (not res) {
    tscl::logger("Controller failed with an exception", tscl::Log::Error);
    tscl::logger(res.getMessage(), tscl::Log::Error);
    return false;
  }
  // nnet::MLPModelSerializer::writeToFile(output_path / "model.nnet", *model);

  return true;
}

void rawTransformation(std::string &input_path, std::string &output_path) {
  // Make input_path and output_path absolute
  input_path = std::filesystem::absolute(input_path);
  output_path = std::filesystem::absolute(output_path);

  if (not std::filesystem::is_directory(input_path)) {
    tscl::logger("Skipping " + input_path + ", not a folder.", tscl::Log::Error);
    return;
  } else {
    tscl::logger("Processing " + input_path, tscl::Log::Debug);
    tscl::logger("Output path: " + output_path, tscl::Log::Debug);
  }

#pragma omp parallel for
  for (auto &entry : std::filesystem::directory_iterator(input_path)) {
    // Get the file name
    std::string file_name = entry.path().filename().string();
    // If it's not a directory
    if (not std::filesystem::is_directory(entry.path())) {
      // Get the extension
      std::string extension = entry.path().extension().string();
      // If it's a jpg
      if (extension == ".jpg" || extension == ".jpeg" || extension == ".png") {
        // Load the image
        auto image = image::ImageSerializer::load(entry.path().string());

        // Transformations
        image::transform::Resize resize(32, 32);
        resize.transform(image);

        image::transform::Inversion inversion;
        inversion.transform(image);

        // Create a new file name
        std::string new_file_name = file_name.substr(0, file_name.size() - 4) + ".png";
        // Create the new file path
        std::string new_file_path = output_path + "/" + new_file_name;
        // Save the image
        tscl::logger("Saving " + new_file_path, tscl::Log::Trace);
        image::ImageSerializer::save(new_file_path, image);
      }
    } else {
      tscl::logger("Skipping " + entry.path().string() + ", is a directory.", tscl::Log::Debug);
      auto sub_input_path = entry.path().string();
      rawTransformation(sub_input_path, output_path);
    }
  }
}

int main(int argc, char **argv) {
  Version::setCurrent(Version(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH, VERSION_TWEAK));
  setupLogger();


  if (argc < 2) {
    tscl::logger("Usage: " + std::string(argv[0]) + " <input_path> (<output_path>)",
                 tscl::Log::Information);
    return 1;
  } else if (argc >= 3) {
    tscl::logger("Using output path: " + std::string(argv[2]), tscl::Log::Information);
    std::string input_path = argv[1];
    std::string output_path = argv[2];
    rawTransformation(input_path, output_path);
    return 0;
  }

  tscl::logger("Initializing OpenCL...", tscl::Log::Debug);
  // utils::clPlatformSelector::initOpenCL();
  utils::clWrapper::initOpenCL(*utils::clWrapper::makeDefault());

  std::vector<std::string> args;
  for (size_t i = 0; i < argc; i++) args.emplace_back(argv[i]);


  return createAndTrain(args[1], args.size() == 3 ? args[2] : "runs/test");
  // predict_test();
  // return 0;
}