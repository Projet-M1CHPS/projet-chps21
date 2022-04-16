#include "EvalController.hpp"
#include "MPITrainingController.hpp"
#include "ModelEvaluator.hpp"
#include "NeuralNetwork.hpp"
#include "ParallelScheduler.hpp"
#include "ProjectVersion.hpp"
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

// MPI variables
static int win_mutex = -1;
int rank = 0;
int nprocess = 0;

void setupLogger() {
  auto &handler = logger.addHandler<StreamLogHandler>("term", std::cout);
  handler.tsType(tscl::timestamp_t::Partial);
  handler.minLvl(Log::Trace);

  auto &thandler = logger.addHandler<StreamLogHandler>("file", "logs.txt");
  thandler.minLvl(Log::Information);
}

// Printf overload
void mpi_put(const std::string &message, tscl::Log::log_level level = tscl::Log::Debug) {
  std::string msg = message;
  msg.insert(0, "[P" + std::to_string(rank) + "]: ");
  msg.insert(0, rank == 0 ? "\033[0;33m" : "\033[0;35m");
  msg.append("\033[0m");

  MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win_mutex);   // Lock the mutex
  tscl::logger(msg, level);
  MPI_Win_unlock(0, win_mutex);   // Unlock the mutex
}

// Create a single mutex for all the processes, accessible by MPI_Win_lock
void create_output_mutex() {
  if (win_mutex)
    MPI_Win_create(&rank, sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_mutex);
  MPI_Win_fence(0, win_mutex);
  mpi_put("Mutex created in main scope");
}

void initializeMPI(int &rank, int &world_size) {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  create_output_mutex();
}

void receiveTensor(math::clFTensor &tensor, std::vector<size_t> &ids, std::vector<long> class_ids,
                   size_t tensor_index) {
  mpi_put("> receiveTensor(...)");
  MPI_Status status;

  // Receive the tensors dimensions [OFFSET, ROWS, COLS, DEPTH]
  std::vector<unsigned long> tensor_dimensions(4);
  MPI_Recv(tensor_dimensions.data(), 4, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  mpi_put("Received tensor " + std::to_string(tensor_index) + " dimensions: (" +
          std::to_string(tensor_dimensions.at(0)) + ", " + std::to_string(tensor_dimensions.at(1)) +
          ", " + std::to_string(tensor_dimensions.at(2)) + ", " +
          std::to_string(tensor_dimensions.at(3)) + ")");

  // Receive the tensor
  math::clFTensor tmp_tensor(tensor_dimensions.at(1), tensor_dimensions.at(2),
                             tensor_dimensions.at(3));
  math::FloatMatrix tmp_matrix(tensor_dimensions.at(1), tensor_dimensions.at(2));
  for (int z = 0; z < tensor_dimensions.at(3); z++) {
    MPI_Recv(tmp_matrix.getData(), (int) (tensor_dimensions.at(1) * tensor_dimensions.at(2)),
             MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    mpi_put("Received fMatrix " + std::to_string(z));
    tmp_tensor[z] = math::clFMatrix(tmp_matrix);
  }
  mpi_put("Received entire tensor " + std::to_string(tensor_index));

  // Receive the samples ids
  int nb_ids = 0;
  MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
  MPI_Get_count(&status, MPI_UNSIGNED_LONG, &nb_ids);
  ids.resize(nb_ids);
  MPI_Recv(ids.data(), nb_ids, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  mpi_put("Received " + std::to_string(nb_ids) + " ids");

  // Receive the samples class ids
  int nb_class_ids = 0;
  MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
  MPI_Get_count(&status, MPI_UNSIGNED_LONG, &nb_class_ids);
  class_ids.resize(nb_class_ids);
  MPI_Recv(class_ids.data(), nb_class_ids, MPI_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  mpi_put("Received " + std::to_string(nb_class_ids) + " class ids");

  // Todo: assign referenced tensor


  mpi_put("< receiveTensor(...)");
}

void sendTensor(const InputSet &input_set, const math::clFTensor &tensor, int dest) {
  mpi_put("> sendTensor(...)");

  const auto &tensor_buffer = tensor.getBuffer();
  MPI_Request dimension_request;

  // Get tensor dimensions
  size_t depth = tensor.getDepth();
  size_t rows = tensor.getRows();
  size_t cols = tensor.getCols();
  // Create an aligned tuple of dimensions
  std::tuple<unsigned long, unsigned long, unsigned long, unsigned long> dims =
          std::make_tuple(tensor.getOffset(), rows, cols, depth);

  // Send the dimensions [OFFSET, ROWS, COLS, DEPTH]
  mpi_put("Sending tensor dimensions (" + std::to_string(tensor.getOffset()) + "," +
          std::to_string(rows) + "," + std::to_string(cols) + "," + std::to_string(depth) +
          ") to " + std::to_string(dest));
  MPI_Isend(&dims, 4, MPI_UNSIGNED_LONG, dest, 0, MPI_COMM_WORLD, &dimension_request);

  // Wait for the dimensions to be sent
  MPI_Wait(&dimension_request, MPI_STATUS_IGNORE);
  mpi_put("Sent tensor dimensions to " + std::to_string(dest));

  // Send the tensor
  mpi_put("Sending tensor to " + std::to_string(dest));
  for (int z = 0; z < tensor.getMatrices().size(); z++) {
    auto matrix = tensor[z];
    MPI_Send(matrix.toFloatMatrix().getData(), (int) matrix.size(), MPI_FLOAT, dest, 0,
             MPI_COMM_WORLD);
  }
  mpi_put("Sent tensor " + std::to_string(dest) + " to process " + std::to_string(dest));

  // Send samples ids
  auto class_ids = input_set.getSamplesIds();
  mpi_put("Sending samples ids to " + std::to_string(dest));
  MPI_Send(class_ids.data(), (int) class_ids.size(), MPI_UNSIGNED_LONG, dest, 0, MPI_COMM_WORLD);
  mpi_put("Sent class ids to " + std::to_string(dest));

  // Send samples class ids
  auto samples_class_ids = input_set.getSamplesClassIds();
  mpi_put("Sending samples class ids to " + std::to_string(dest));
  MPI_Send(samples_class_ids.data(), (int) samples_class_ids.size(), MPI_LONG, dest, 0,
           MPI_COMM_WORLD);
  mpi_put("Sent samples class ids to " + std::to_string(dest));

  mpi_put("< sendTensor(...)");
}

void scatterTrainingCollections(const std::vector<TrainingCollection> &send_collections,
                                TrainingCollection &recv_collection) {
  if (rank == 0) {
    for (int p = 0; p < nprocess; p++) {
      const TrainingCollection &current_collection = send_collections.at(p);
      assert(!current_collection.getTargets().empty());

      MPI_Send((const void *) current_collection.getTargets().size(), 1, MPI_UNSIGNED_LONG, p, 0,
               MPI_COMM_WORLD);

      for (size_t t = 0; t < current_collection.getTargets().size(); t++)
        sendTensor(current_collection.getTrainingSet(), current_collection.getTargets().at(t), p);
    }
  } else {
    unsigned long nb_targets = 0;
    MPI_Recv(&nb_targets, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    mpi_put("Received " + std::to_string(nb_targets) + ", the targets count, from process 0");
    std::vector<math::clFTensor> targets(nb_targets);
    std::vector<size_t> ids;
    std::vector<long> class_ids;

    for (size_t t = 0; t < nb_targets; t++) receiveTensor(targets.at(t), ids, class_ids, t);
  }
}

// This function is quite big, but it allows the user to trivially specify the hyperparameters in
// one place. This is especially useful for debugging or benchmarking. This function is not intended
// for production use.
bool createAndTrain(std::filesystem::path const &input_path,
                    std::filesystem::path const &output_path) {
  tscl::logger("Current version: " + tscl::Version::current.to_string(), tscl::Log::Debug);
  tscl::logger("Output path: " + output_path.string(), tscl::Log::Debug);

  if (not std::filesystem::exists(output_path)) std::filesystem::create_directories(output_path);

  // MPI shared variables
  TrainingCollection training_collection;
  std::vector<TrainingCollection> training_collections;

  // Change this for benchmarking
  // One day, we'll have enough time to make this configurable from a file :c
  constexpr int kImageSize = 32;   // We assume we use square images
  // Size to use for images allocation
  constexpr int kTensorSize = 512;
  // The size of the batch. We highly recommend using a multiple/dividend of the tensor size
  // to avoid batch fragmentation
  constexpr int kBatchSize = 16;

  constexpr float kLearningRate = 0.01;
  constexpr float kMomentum = 0.9;
  constexpr float kDecayRate = 0.001;

  enum OptimType { kUseSGD, kUseMomentum, kUseDecay, kUseDecayMomentum };

  constexpr OptimType kOptimType = kUseSGD;

  // Maximum number of thread
  // The scheduler is free to use less if it judges necessary
  constexpr size_t kMaxThread = 1;
  constexpr bool kAllowMultipleThreadPerDevice = false;
  constexpr size_t kMaxEpoch = 100;
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

  // MPI Master section
  {
    if (rank == 0) {
      mpi_put("Loading dataset...", tscl::Log::Information);
      TrainingCollectionLoader loader(kTensorSize, kImageSize, kImageSize);
      auto &pre_engine = loader.getPreProcessEngine();
      // Add preprocessing transformations here
      pre_engine.addTransformation(std::make_shared<image::transform::Inversion>());

      auto &engine = loader.getPostProcessEngine();
      // Add postprocessing transformations here
      engine.addTransformation(std::make_shared<image::transform::BinaryScale>());

      TrainingCollection full_training_collection = loader.load(input_path);
      mpi_put("Dataset loaded", tscl::Log::Information);

      full_training_collection.display();

      // Split the dataset into nprocess parts
      mpi_put("Splitting dataset...", tscl::Log::Information);
      training_collections.reserve(nprocess);
      training_collections = full_training_collection.split(nprocess);
      mpi_put("Dataset split", tscl::Log::Information);

      for (auto &item : training_collections) item.display();
    }
  }

  return true;

  mpi_put("Scattering dataset...", tscl::Log::Information);
  scatterTrainingCollections(training_collections, training_collection);
  mpi_put("Dataset scattered", tscl::Log::Information);

  return true;

  topology.pushBack(training_collection.getClassCount());

  logger("Training set size: " + std::to_string(training_collection.getTrainingSet().getSize()),
         tscl::Log::Trace);
  logger("Testing set size: " + std::to_string(training_collection.getEvaluationSet().getSize()),
         tscl::Log::Trace);


  // auto model = nnet::MLPModel::randomReluSigmoid(topology);
  auto model = nnet::MLPModel::random(topology, af::ActivationFunctionType::leakyRelu);
  // auto model = std::make_unique<nnet::MLPModel>();
  // model->load("michal.nnet");
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

  // ParallelScheduler scheduler(batch, input, target);

  logger("Creating scheduler", tscl::Log::Debug);

  ParallelScheduler::Builder scheduler_builder;
  scheduler_builder.setTrainingSets(training_collection.getTrainingSet().getTensors(),
                                    training_collection.getTargets());
  // Set the resources for the scheduler
  scheduler_builder.setMaxThread(kMaxThread, kAllowMultipleThreadPerDevice);
  scheduler_builder.setDevices(utils::cl_wrapper.getDevices());

  scheduler_builder.setOptimizer(*optimizer);

  // Set the batch size, and allow/disallow batch optimization
  scheduler_builder.setBatchSize(kBatchSize, kAllowBatchDefragmentation);

  auto scheduler = scheduler_builder.build();
  // SchedulerProfiler sc_profiler(scheduler_builder.build(), output_path / "scheduler");
  //  sc_profiler.setVerbose(false);

  ModelEvolutionTracker evaluator(output_path / "model_evolution", *model, training_collection);

  logger("Starting run", tscl::Log::Debug);
  MPITrainingController controller(kMaxEpoch, evaluator, *scheduler);
  controller.setVerbose(true);
  ControllerResult res = controller.run();
  // Ensure the profiler dumps to disk cleanly
  // sc_profiler.finish();

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

  MPI_Init(&argc, &argv);
  initializeMPI(rank, nprocess);
  auto ret = createAndTrain(args[1], args.size() == 3 ? args[2] : "runs/test");
  MPI_Finalize();
  return ret;
}
