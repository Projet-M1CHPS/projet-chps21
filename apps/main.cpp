#include "Control.hpp"
#include "NeuralNetwork.hpp"
#include "ProjectVersion.hpp"
#include "tscl.hpp"

#include <iomanip>
#include <iostream>
#include <vector>

#include <CL/opencl.hpp>

using namespace control;
using namespace control::classifier;
using namespace tscl;

void setupLogger() {
  auto &handler = logger.addHandler<StreamLogHandler>("term", std::cout);
  handler.tsType(tscl::timestamp_t::Partial);
  handler.minLvl(Log::Trace);
  // handler.enable(false);

  auto &thandler = logger.addHandler<StreamLogHandler>("file", "logs.txt");
  thandler.minLvl(Log::Information);
}

bool createAndTrain(std::shared_ptr<utils::clWrapper> wrapper,
                    std::filesystem::path const &input_path,
                    std::filesystem::path const &output_path) {
  tscl::logger("Current version: " + tscl::Version::current.to_string(), tscl::Log::Debug);
  tscl::logger("Fetching model from  " + input_path.string(), tscl::Log::Debug);
  tscl::logger("Output path: " + output_path.string(), tscl::Log::Debug);

  if (not std::filesystem::exists(output_path)) std::filesystem::create_directories(output_path);

  tscl::logger("Creating collection loader", tscl::Log::Debug);
  CITCLoader loader(32, 32);
  auto &pre_engine = loader.getPreProcessEngine();
  // Add preprocessing transformations here
  pre_engine.addTransformation(std::make_shared<image::transform::Inversion>());
  auto &engine = loader.getPostProcessEngine();
  // Add postprocessing transformations here
  // engine.addTransformation(std::make_shared<image::transform::BinaryScale>());

  tscl::logger("Loading collection", tscl::Log::Information);
  auto training_collection = loader.load(input_path, *wrapper);


  // Create a correctly-sized topology
  nnet::MLPTopology topology = {32 * 32, 64, 64, 32, 32};
  topology.pushBack(training_collection->getClassCount());

  auto model = nnet::MLPModel::randomReluSigmoid(wrapper, topology);

  exit(1);
  auto optimizer = nnet::MLPMiniBatchOptimizer::make<nnet::DecayMomentumOptimization>(
          *model, 64, 0.2, 0.1, 0.9);

  tscl::logger("Creating controller", tscl::Log::Trace);

  TrainingControllerParameters parameters(input_path, output_path, 100, 1, false);
  CTController controller(parameters, *model, *optimizer, *training_collection);
  ControllerResult res = controller.run();

  if (not res) {
    tscl::logger("Controller failed with an exception", tscl::Log::Error);
    tscl::logger(res.getMessage(), tscl::Log::Error);
    return false;
  }
  nnet::MLPModelSerializer::writeToFile(output_path / "model.nnet", *model);
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
  std::shared_ptr<utils::clWrapper> wrapper = utils::clWrapper::makeDefaultWrapper();

  std::vector<std::string> args;
  for (size_t i = 0; i < argc; i++) args.emplace_back(argv[i]);

  return createAndTrain(wrapper, args[1], args.size() == 3 ? args[2] : "runs/test");
}