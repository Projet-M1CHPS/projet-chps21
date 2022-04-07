#include "EvalController.hpp"
#include "NeuralNetwork.hpp"
#include "ProjectVersion.hpp"
#include "TrainingController.hpp"
#include "clUtils/clPlatformSelector.hpp"
#include "controlSystem2/TrainingCollection.hpp"
#include "controlSystem2/TrainingCollectionLoader.hpp"
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

  constexpr int kImageSize = 32;
  // Ensure this is the same size as the batch size
  constexpr int kTensorSize = 256;

  tscl::logger("Loading dataset", tscl::Log::Debug);
  TrainingCollectionLoader loader(kTensorSize, kImageSize, kImageSize);
  auto &pre_engine = loader.getPreProcessEngine();
  // Add preprocessing transformations here
  pre_engine.addTransformation(std::make_shared<image::transform::Inversion>());

  auto &engine = loader.getPostProcessEngine();
  // Add postprocessing transformations here
  // engine.addTransformation(std::make_shared<image::transform::BinaryScale>());

  TrainingCollection training_collection = loader.load(input_path);

  tscl::logger("Training set size: " +
                       std::to_string(training_collection.getTrainingSet().getSize()),
               tscl::Log::Trace);
  tscl::logger("Testing set size: " +
                       std::to_string(training_collection.getEvaluationSet().getSize()),
               tscl::Log::Trace);


  // Create a correctly-sized topology
  nnet::MLPTopology topology = {kImageSize * kImageSize, 64, 64, 64, 64, 64, 64};
  topology.pushBack(training_collection.getClassCount());

  auto model = nnet::MLPModel::randomReluSigmoid(topology);

  auto optimizer = nnet::MLPStochOptimizer::make<nnet::SGDOptimization>(*model, 0.08);
  // auto optimizer = nnet::MLPBatchOptimizer::make<nnet::SGDOptimization>(*model, 0.03);

  tscl::logger("Creating controller", tscl::Log::Trace);
  // EvalController controller(output_path, model.get(), &training_collection.getEvaluationSet());
  TrainingController controller(output_path, *model, *optimizer, training_collection, 100);
  ControllerResult res = controller.run();

  if (not res) {
    tscl::logger("Controller failed with an exception", tscl::Log::Error);
    tscl::logger(res.getMessage(), tscl::Log::Error);
    return false;
  }
  // nnet::MLPModelSerializer::writeToFile(output_path / "model.nnet", *model);

  return true;
}

int main(int argc, char **argv) {
  Version::setCurrent(Version(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH, VERSION_TWEAK));
  setupLogger();


  if (argc < 2) {
    tscl::logger("Usage: " + std::string(argv[0]) + " <input_path> (<output_path>)",
                 tscl::Log::Information);
    return 1;
  }

  tscl::logger("Initializing OpenCL...", tscl::Log::Debug);
  // utils::clPlatformSelector::initOpenCL();
  utils::clWrapper::initOpenCL(*utils::clWrapper::makeDefault());

  std::vector<std::string> args;
  for (size_t i = 0; i < argc; i++) args.emplace_back(argv[i]);


  return createAndTrain(args[1], args.size() == 3 ? args[2] : "runs/test");
}