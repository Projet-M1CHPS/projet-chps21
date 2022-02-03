#include "Control.hpp"
#include "Network.hpp"
#include "ProjectVersion.hpp"
#include "tscl.hpp"

#include <iomanip>
#include <iostream>
#include <vector>

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

bool createAndTrain(std::filesystem::path const &input_path,
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
  engine.addTransformation(std::make_shared<image::transform::BinaryScale>());

  tscl::logger("Loading collection", tscl::Log::Information);
  auto training_collection = loader.load(input_path, wrapper);

  // Create a correctly-sized topology
  nnet::MLPTopology topology = {32 * 32, 64, 64, 32, 32};
  topology.push_back(training_collection->getClassCount());

  auto model = nnet::MLPModelFactory::randomSigReluAlt(topology);

  auto tm =
          std::make_shared<nnet::DecayMomentumOptimization>(model->getPerceptron(), 0.2, 0.1, 0.9);

  auto optimizer = std::make_unique<nnet::MLPMiniBatchOptimizer>(*model, tm, 16);

  tscl::logger("Creating controller", tscl::Log::Trace);

  TrainingControllerParameters parameters(input_path, output_path, 100, 1, false);
  CTController controller(parameters, *model, *optimizer, *training_collection);
  ControllerResult res = controller.run();

  if (not res) {
    tscl::logger("Controller failed with an exception", tscl::Log::Error);
    tscl::logger(res.getMessage(), tscl::Log::Error);
    return false;
  }
  nnet::PlainTextMLPModelSerializer serializer;
  serializer.writeToFile(output_path / "model.nnet", *model);
  return true;
}

#include <CL/opencl.hpp>

int main(int argc, char **argv) {
  Version::setCurrent(Version(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH, VERSION_TWEAK));
  setupLogger();

  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);
  if (all_platforms.empty()) {
    tscl::logger("No OpenCL platforms found", tscl::Log::Error);
    return 1;
  }

  for (auto &platform : all_platforms) {
    tscl::logger("Found platform: " + platform.getInfo<CL_PLATFORM_NAME>(), tscl::Log::Debug);
  }
  cl::Platform default_platform = all_platforms[0];
  tscl::logger("Using platform: " + default_platform.getInfo<CL_PLATFORM_NAME>(), tscl::Log::Debug);

  std::vector<cl::Device> all_devices;
  default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);

  if (all_devices.empty()) {
    tscl::logger("No OpenCL devices found", tscl::Log::Error);
    return 1;
  }
  cl::Device default_device = all_devices[0];
  tscl::logger("Using device: " + default_device.getInfo<CL_DEVICE_NAME>(), tscl::Log::Debug);
  cl::Context context(all_devices);
  cl::Context::setDefault(context);
  cl::CommandQueue queue(context, default_device);
  cl::CommandQueue::setDefault(queue);


  std::vector<std::string> args;
  for (size_t i = 0; i < argc; i++) args.emplace_back(argv[i]);

  return createAndTrain(args[1], args.size() == 3 ? args[2] : "runs/test");
}