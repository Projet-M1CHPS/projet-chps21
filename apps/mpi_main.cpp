#include "EvalController.hpp"
#include "MPINeuralNetwork.hpp"
#include "MPIParallelScheduler.hpp"
#include "MPITrainingController.hpp"
#include "ModelEvaluator.hpp"
#include "ProjectVersion.hpp"
#include "controlSystem/TrainingCollection.hpp"
#include "controlSystem/TrainingCollectionLoader.hpp"
#include "mpiWrapper/TrainingCollectionScatterer.hpp"
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
  constexpr int kTensorSize = 250;
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
  constexpr size_t kMaxThread = 1;
  constexpr bool kAllowMultipleThreadPerDevice = false;
  constexpr size_t kMaxEpoch = 75;
  // If set to true, the scheduler will move batches around to ensure each batch used for
  // computation is of size kBatchSize
  constexpr bool kAllowBatchDefragmentation = false;
  constexpr size_t kMaxDeviceCount = 1;

  // utils::clPlatformSelector::initOpenCL();
  // Uncomment to display a ncurses-based UI for platform selection
  // Take care to only call initOpenCL ONCE!
  utils::clWrapper::initOpenCL(*utils::clWrapper::makeDefault());
  std::vector<cl::Device> allowed_devices = utils::cl_wrapper.getDevices();

  // Just truncate the list of devices to kMaxDeviceCount (We assume every device is the same for
  // the sake of simplicity)
  allowed_devices.resize(kMaxDeviceCount);
  // This is pretty bad design, but it'll simplify the benchmarking process
  // Not intended for production code
  utils::cl_wrapper.restrictDevicesTo(allowed_devices);

  // Do not add the output size, it is automatically set to the number of classes
  MLPTopology topology = {kImageSize * kImageSize, 64, 64, 64, 64};

  int rank = 0;
  int comm_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  mpiw::TrainingCollectionScatterer scatterer(MPI_COMM_WORLD);
  std::unique_ptr<TrainingCollection> local_collection;
  if (rank == 0) {
    TrainingCollectionLoader loader(kTensorSize, kImageSize, kImageSize);
    auto &pre_engine = loader.getPreProcessEngine();
    // Add preprocessing transformations here
    pre_engine.addTransformation(std::make_shared<image::transform::Inversion>());

    auto &engine = loader.getPostProcessEngine();
    // Add postprocessing transformations here
    // engine.addTransformation(std::make_shared<image::transform::BinaryScale>());

    TrainingCollection full_training_collection = loader.load(input_path);
    local_collection =
            std::make_unique<TrainingCollection>(scatterer.scatter(full_training_collection));
    local_collection->getEvaluationSet() = std::move(full_training_collection.getEvaluationSet());
  } else {
    local_collection = std::make_unique<TrainingCollection>(scatterer.receive(0));
  }

  std::filesystem::create_directories("tmp_" + std::to_string(rank));
  std::cout << rank << " Saving to "
            << "tmp_" + std::to_string(rank)
            << " count: " << local_collection->getTrainingSet().getSize() << std::endl;

  topology.pushBack(local_collection->getClassCount());

  if (rank == 0) {
    logger("Training set size: " + std::to_string(local_collection->getTrainingSet().getSize()),
           tscl::Log::Trace);
    logger("Evaluation set size: " + std::to_string(local_collection->getEvaluationSet().getSize()),
           tscl::Log::Trace);
  }

  // auto model = nnet::MLPModel::randomReluSigmoid(topology);
  auto model = nnet::MPIMLPModel::random(topology, af::ActivationFunctionType::leakyRelu);
  // auto model = std::make_unique<nnet::MLPModel>();
  // model->load("michal.nnet");
  std::unique_ptr<Optimizer> optimizer;
  if (kOptimType == kUseSGD)
    optimizer = nnet::MPIMLPOptimizer::make<nnet::SGDOptimization>(*model, kLearningRate);
  else if (kOptimType == kUseMomentum)
    optimizer = nnet::MPIMLPOptimizer::make<nnet::MomentumOptimization>(*model, kLearningRate,
                                                                        kMomentum);
  else if (kOptimType == kUseDecay)
    optimizer =
            nnet::MPIMLPOptimizer::make<nnet::DecayOptimization>(*model, kLearningRate, kDecayRate);
  else if (kOptimType == kUseDecayMomentum)
    optimizer = nnet::MPIMLPOptimizer::make<nnet::DecayMomentumOptimization>(*model, kLearningRate,
                                                                             kDecayRate, kMomentum);
  logger("[P" + std::to_string(rank) + "]: " + "Creating scheduler", tscl::Log::Debug);


  MPIParallelScheduler::Builder scheduler_builder;
  std::cout << "batch size : " << static_cast<size_t>(kBatchSize / comm_size) << std::endl;
  auto targets = local_collection->makeTargets();
  scheduler_builder.setJob({static_cast<size_t>(kBatchSize / comm_size),
                            local_collection->getTrainingSet().getTensors(), targets});
  // Set the resources for the scheduler
  scheduler_builder.setMaxThread(kMaxThread, kAllowMultipleThreadPerDevice);
  scheduler_builder.setDevices(utils::cl_wrapper.getDevices());

  scheduler_builder.setOptimizer(*optimizer);

  auto scheduler = scheduler_builder.build();
  // SchedulerProfiler sc_profiler(scheduler_builder.build(), output_path / "scheduler");
  //  sc_profiler.setVerbose(false);

  ModelEvolutionTracker evaluator(output_path / ("model_evolution_" + std::to_string(rank)), *model,
                                  *local_collection);

  logger("[P" + std::to_string(rank) + "]: Starting run", tscl::Log::Debug);
  MPITrainingController controller(kMaxEpoch, evaluator, *scheduler);
  controller.setVerbose(true);

  ControllerResult res = controller.run();
  // Ensure the profiler dumps to disk cleanly
  // sc_profiler.finish();

  if (not res) {
    logger("[P" + std::to_string(rank) + "]: Controller failed with an exception",
           tscl::Log::Error);
    logger("[P" + std::to_string(rank) + "]: " + res.getMessage(), tscl::Log::Error);
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

  MPI_Init(nullptr, nullptr);
  int n_process = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &n_process);

  auto ret = createAndTrain(args[1], args.size() == 3 ? args[2] : "runs/test");
  MPI_Finalize();
  return ret;
}
