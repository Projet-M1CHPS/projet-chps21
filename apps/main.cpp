#include "EvalController.hpp"
#include "ModelEvaluator.hpp"
#include "NeuralNetwork.hpp"
#include "ParallelScheduler.hpp"
#include "ProjectVersion.hpp"
#include "TrainingController.hpp"
#include "controlSystem/TrainingCollection.hpp"
#include "controlSystem/TrainingCollectionLoader.hpp"
#include "neuralNetwork/OptimizationScheduler/SchedulerProfiler.hpp"
#include "openclUtils/clPlatformSelector.hpp"
#include "tscl.hpp"

#include <iomanip>
#include <iostream>
#include <vector>

using namespace control;
using namespace tscl;
using namespace nnet;

void setupLogger() {
  auto &handler = logger.addHandler<StreamLogHandler>("term", std::cout);
  handler.tsType(tscl::timestamp_t::Partial);
  handler.minLvl(Log::Trace);

  auto &thandler = logger.addHandler<StreamLogHandler>("file", "logs.txt");
  thandler.minLvl(Log::Information);
}

// This function is quite big, but it allows the user to trivially specify the hyperparameters in
// one place. This is especially useful for debugging or benchmarking. This function is not intended
// for production use.
bool createAndTrain(std::filesystem::path const &input_path,
                    std::filesystem::path const &output_path) {
  tscl::logger("Current version: " + tscl::Version::current.to_string(), tscl::Log::Debug);
  tscl::logger("Output path: " + output_path.string(), tscl::Log::Debug);

  if (not std::filesystem::exists(output_path)) std::filesystem::create_directories(output_path);

  // Change this for benchmarking
  // One day, we'll have enough time to make this configurable from a file :c
  constexpr int kImageSize = 32;   // We assume we use square images
  // Size to use for images allocation
  constexpr int kTensorSize = 256;
  // The size of the batch. We highly recommend using a multiple/dividend of the tensor size
  // to avoid batch fragmentation
  constexpr int kBatchSize = 16;

  constexpr float kLearningRate = 0.08;
  constexpr float kMomentum = 0.9;
  constexpr float kDecayRate = 0.001;

  enum OptimType { kUseSGD, kUseMomentum, kUseDecay, kUseDecayMomentum };

  constexpr OptimType kOptimType = kUseSGD;

  // Maximum number of thread
  // The scheduler is free to use less if it judges necessary
  constexpr size_t kMaxThread = 4;
  constexpr bool kAllowMultipleThreadPerDevice = false;
  constexpr size_t kMaxEpoch = 75;
  // If set to true, the scheduler will move batches around to ensure each batch used for
  // computation is of size kBatchSize
  constexpr size_t kMaxDeviceCount = 4;

  // utils::clPlatformSelector::initOpenCL();
  // Uncomment to display a ncurses-based UI for platform selection
  // Take care to only call initOpenCL ONCE!
  utils::clWrapper::initOpenCL(*utils::clWrapper::makeDefault());
  std::vector<cl::Device> allowed_devices = utils::cl_wrapper.getDevices();

  // Just truncate the list of devices to kMaxDeviceCount (We assume every device is the same for
  // the sake of simplicity)
  allowed_devices.resize(std::min(kMaxDeviceCount, allowed_devices.size()));
  // This is pretty bad design, but it'll simplify the benchmarking process
  // Not intended for production code
  utils::cl_wrapper.restrictDevicesTo(allowed_devices);

  // Do not add the output size, it is automatically set to the number of classes
  MLPTopology topology = {kImageSize * kImageSize, 1024, 1024, 1024, 1024, 512, 256};

  logger("Loading dataset", tscl::Log::Debug);
  TrainingCollectionLoader loader(kTensorSize, kImageSize, kImageSize);
  auto &pre_engine = loader.getPreProcessEngine();
  // Add preprocessing transformations here
  pre_engine.addTransformation(std::make_shared<image::transform::Inversion>());

  auto &engine = loader.getPostProcessEngine();
  // Add postprocessing transformations here
  engine.addTransformation(std::make_shared<image::transform::BinaryScale>());

  TrainingCollection training_collection = loader.load(input_path);
  // Create a correctly-sized output layer
  topology.pushBack(training_collection.getClassCount());


  logger("Training set size: " + std::to_string(training_collection.getTrainingSet().getSize()),
         tscl::Log::Trace);
  logger("Testing set size: " + std::to_string(training_collection.getEvaluationSet().getSize()),
         tscl::Log::Trace);


  // auto model = nnet::MLPModel::randomReluSigmoid(topology);
  auto model = nnet::MLPModel::random(topology, af::ActivationFunctionType::leakyRelu);
  /*auto model = std::make_unique<nnet::MLPModel>();
  model->load("michal.nnet");*/
  std::unique_ptr<Optimizer> optimizer;
  if (kOptimType == kUseSGD)
    optimizer = nnet::MLPOptimizer::make<nnet::SGDOptimization>(*model, kLearningRate);
  else if (kOptimType == kUseMomentum)
    optimizer =
            nnet::MLPOptimizer::make<nnet::MomentumOptimization>(*model, kLearningRate, kMomentum);
  else if (kOptimType == kUseDecay)
    optimizer =
            nnet::MLPOptimizer::make<nnet::DecayOptimization>(*model, kLearningRate, kDecayRate);
  else if (kOptimType == kUseDecayMomentum)
    optimizer = nnet::MLPOptimizer::make<nnet::DecayMomentumOptimization>(*model, kLearningRate,
                                                                          kDecayRate, kMomentum);


  logger("Creating scheduler", tscl::Log::Debug);

  ParallelScheduler::Builder scheduler_builder;
  scheduler_builder.setJob({kBatchSize, training_collection.getTrainingSet().getTensors(),
                            training_collection.getTargets()});
  // Set the resources for the scheduler
  scheduler_builder.setMaxThread(kMaxThread, kAllowMultipleThreadPerDevice);
  scheduler_builder.setDevices(utils::cl_wrapper.getDevices());

  scheduler_builder.setOptimizer(*optimizer);

  auto scheduler = scheduler_builder.build();
  // SchedulerProfiler sc_profiler(scheduler_builder.build(), output_path / "scheduler");
  // sc_profiler.setVerbose(false);

  ModelEvolutionTracker evaluator(output_path / "model_evolution", *model, training_collection);

  logger("Starting run", tscl::Log::Debug);
  TrainingController controller(kMaxEpoch, evaluator, *scheduler);
  controller.setVerbose(true);
  ControllerResult res = controller.run();

  if (not res) {
    tscl::logger("Controller failed with an exception", tscl::Log::Error);
    tscl::logger(res.getMessage(), tscl::Log::Error);
    return false;
  }
  return true;
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
  }

  tscl::logger("Initializing OpenCL...", tscl::Log::Debug);

  std::vector<std::string> args;
  for (size_t i = 0; i < argc; i++) args.emplace_back(argv[i]);


  return createAndTrain(args[1], args.size() == 3 ? args[2] : "runs/test");
}