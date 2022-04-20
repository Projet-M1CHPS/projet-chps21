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

  MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, win_mutex);   // Lock the mutex
  tscl::logger(msg, level);
  MPI_Win_unlock(rank, win_mutex);   // Unlock the mutex
}

// Create a single mutex for all the processes, accessible by MPI_Win_lock
void create_output_mutex() {
  if (win_mutex)
    MPI_Win_create(&rank, sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_mutex);
  MPI_Win_fence(0, win_mutex);
  mpi_put("Mutex created in main scope");
}

void initializeMPI() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocess);
  create_output_mutex();
}

void receiveTensor(math::clFTensor &tensor, std::string &class_name, std::vector<size_t> &ids,
                   std::vector<long> &class_ids, size_t tensor_index) {
  mpi_put("> receiveTensor(...)");
  MPI_Status status;

  // Receive the tensors dimensions [OFFSET, ROWS, COLS, DEPTH]
  std::array<unsigned long, 4> tensor_dimensions{};
  MPI_Recv(tensor_dimensions.data(), 4, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, &status);
  mpi_put("Received tensor " + std::to_string(tensor_index) + " dimensions: (" +
          std::to_string(tensor_dimensions[0]) + ", " + std::to_string(tensor_dimensions[1]) +
          ", " + std::to_string(tensor_dimensions[2]) + ", " +
          std::to_string(tensor_dimensions[3]) + ")");
  size_t tensor_depth = tensor_dimensions[3];

  // Receive the tensor
  math::clFTensor tmp_tensor(tensor_dimensions.at(1), tensor_dimensions.at(2),
                             tensor_dimensions.at(3));

  math::FloatMatrix tmp_matrix(tensor_dimensions.at(1), tensor_dimensions.at(2));
  for (int z = 0; z < tensor_depth; z++) {
    MPI_Recv(tmp_matrix.getData(), (int) (tensor_dimensions.at(1) * tensor_dimensions.at(2)),
             MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    tmp_tensor[z] = math::clFMatrix(tmp_matrix, true);
  }
  mpi_put("Received entire tensor " + std::to_string(tensor_index));

  // Copy the temporary tensor to the final one
  /*cl::CommandQueue queue(cl::Context::getDefault(), cl::Device::getDefault());
  tmp_tensor.copy(tensor, queue, true);*/

  tensor = std::move(tmp_tensor);

  // Receive the samples ids
  ids.resize(tensor_depth);
  MPI_Recv(ids.data(), (int) ids.size(), MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  mpi_put("Received " + std::to_string(tensor.getDepth()) + " ids");

  // Receive the samples class ids
  class_ids.resize(tensor_depth);
  MPI_Recv(class_ids.data(), (int) class_ids.size(), MPI_LONG, 0, 0, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  mpi_put("Received " + std::to_string(tensor.getDepth()) + " class ids");

  // Receive class_name
  MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
  int class_name_length;
  MPI_Get_count(&status, MPI_CHAR, &class_name_length);
  class_name.resize(class_name_length);
  MPI_Recv(class_name.data(), (int) class_name.size(), MPI_CHAR, 0, 0, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  mpi_put("Received class name: " + class_name);

  mpi_put("< receiveTensor(...)");
}

void sendTensor(const InputSet &input_set, const std::string &class_name,
                const math::clFTensor &tensor, int dest, size_t tensor_index) {
  mpi_put("> sendTensor(...)");
  size_t tensor_class_offset = 0;
  for (size_t i = 0; i < tensor_index; i++)
    tensor_class_offset += input_set.getTensor(i).getDepth();

  MPI_Request dimension_request;

  // Get tensor dimensions
  size_t depth = tensor.getDepth();
  size_t rows = tensor.getRows();
  size_t cols = tensor.getCols();
  // Create an aligned tuple of dimensions
  std::array<unsigned long, 4> dims{tensor.getOffset(), rows, cols, depth};

  // Send the dimensions [OFFSET, ROWS, COLS, DEPTH]
  mpi_put("Sending tensor dimensions (" + std::to_string(tensor.getOffset()) + "," +
          std::to_string(rows) + "," + std::to_string(cols) + "," + std::to_string(depth) +
          ") to " + std::to_string(dest));
  MPI_Send(&dims, 4, MPI_UNSIGNED_LONG, dest, 0, MPI_COMM_WORLD);

  // Wait for the dimensions to be sent
  // MPI_Wait(&dimension_request, MPI_STATUS_IGNORE);
  mpi_put("Sent tensor dimensions to " + std::to_string(dest));

  assert(depth == tensor.getMatrices().size());

  auto queue = cl::CommandQueue::getDefault();
  // Transfer the buffer to the device with a map


  // Send the tensor
  mpi_put("Sending tensor to " + std::to_string(dest));
  for (int z = 0; z < tensor.getMatrices().size(); z++)
    MPI_Send(tensor[z].toFloatMatrix().getData(), (int) tensor[z].size(), MPI_FLOAT, dest, 0,
             MPI_COMM_WORLD);
  mpi_put("Sent tensor " + std::to_string(dest) + " to process " + std::to_string(dest));

  // Send samples ids
  std::vector<unsigned long> class_ids(tensor.getDepth());
  for (size_t i = 0; i < tensor.getDepth(); i++)
    class_ids[i] = input_set.getSamplesIds().at(tensor_class_offset + i);
  mpi_put("Sending samples ids to " + std::to_string(dest));
  MPI_Send(class_ids.data(), (int) class_ids.size(), MPI_UNSIGNED_LONG, dest, 0, MPI_COMM_WORLD);
  mpi_put("Sent class ids to " + std::to_string(dest));

  // Send samples class ids
  std::vector<long> samples_class_ids(tensor.getDepth());
  for (size_t i = 0; i < tensor.getDepth(); i++)
    samples_class_ids[i] = input_set.getSamplesClassIds().at(tensor_class_offset + i);
  mpi_put("Sending samples class ids to " + std::to_string(dest));
  MPI_Send(samples_class_ids.data(), (int) samples_class_ids.size(), MPI_LONG, dest, 0,
           MPI_COMM_WORLD);
  mpi_put("Sent samples class ids to " + std::to_string(dest));

  mpi_put("Sending class name " + class_name + " to " + std::to_string(dest));
  MPI_Send(class_name.c_str(), (int) class_name.size(), MPI_CHAR, dest, 0, MPI_COMM_WORLD);
  mpi_put("Sent class name to " + std::to_string(dest));

  mpi_put("< sendTensor(...)");
}

TrainingCollection scatterTrainingCollections(std::vector<TrainingCollection> &send_collections) {
  std::array<unsigned long, 2> collection_dim{};
  if (rank == 0) {
    auto &set = send_collections.at(0).getTrainingSet();
    collection_dim = {set.getInputWidth(), set.getInputHeight()};
    assert(std::all_of(collection_dim.cbegin(), collection_dim.cend(),
                       [](const auto &e) { return e > 0; }));
    mpi_put("Sending collection dimensions: " + std::to_string(collection_dim.at(0)) + "x" +
            std::to_string(collection_dim.at(1)));
    for (int p = 1; p < nprocess; p++)
      MPI_Send(collection_dim.data(), 2, MPI_UNSIGNED_LONG, p, 0, MPI_COMM_WORLD);
  } else {
    MPI_Recv(collection_dim.data(), 2, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  TrainingCollection recv_collection(collection_dim.at(0), collection_dim.at(1));

  mpi_put("Recv_collection.dimensions: " + std::to_string(collection_dim.at(0)) + "x" +
          std::to_string(collection_dim.at(1)));

  if (rank == 0) {
    for (int p = 1; p < nprocess; p++) {
      const TrainingCollection &current_collection = send_collections.at(p);
      assert(!current_collection.getTargets().empty());

      // Send the collection size
      mpi_put("Sending collection size (" + std::to_string(current_collection.getTargets().size()) +
              ") to " + std::to_string(p));
      const unsigned long current_targets_count = current_collection.getTargets().size();
      MPI_Send(&current_targets_count, 1, MPI_UNSIGNED_LONG, p, 0, MPI_COMM_WORLD);

      for (size_t t = 0; t < current_collection.getTrainingSet().getTensorCount(); t++)
        sendTensor(current_collection.getTrainingSet(),
                   current_collection.getTrainingSet().getClasses().at(t),
                   current_collection.getTrainingSet().getTensor(t), p, t);
    }
  } else {
    unsigned long nb_targets = 0;
    MPI_Recv(&nb_targets, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    mpi_put("Received " + std::to_string(nb_targets) + ", the targets count, from process 0");


    std::vector<size_t> ids;
    std::vector<long> class_ids;
    math::clFTensor tensor;
    std::vector<std::string> class_names(nb_targets);

    for (size_t t = 0; t < nb_targets; t++) {
      receiveTensor(tensor, class_names.at(t), ids, class_ids, t);
      mpi_put("Received tensor " + std::to_string(t) + " from process 0");
      mpi_put("Received tensor ids: " + std::to_string(ids.size()));
      mpi_put("Received tensor class ids: " + std::to_string(class_ids.size()));
      recv_collection.getTrainingSet().append(std::move(tensor), ids, class_ids);
    }
    recv_collection.getTrainingSet().updateClasses(class_names);
    recv_collection.makeTrainingTargets();
  }

  if (rank == 0) recv_collection = std::move(send_collections.at(0));
  return recv_collection;
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
      full_training_collection.split(nprocess, training_collections);
      mpi_put("Dataset split", tscl::Log::Information);
    }
  }


  // Repartition the training collections
  mpi_put("Scattering dataset...", tscl::Log::Information);
  TrainingCollection local_training_collection = scatterTrainingCollections(training_collections);
  mpi_put("Dataset scattered", tscl::Log::Information);

  assert(local_training_collection.getTrainingSet().getTensorCount() > 0);
  assert(!local_training_collection.getTrainingSet().getSamplesClassIds().empty());
  assert(!local_training_collection.getTrainingSet().getSamplesIds().empty());
  assert(local_training_collection.getTrainingSet().getClassCount() > 0);


  topology.pushBack(local_training_collection.getClassCount());

  logger("Training set size: " +
                 std::to_string(local_training_collection.getTrainingSet().getSize()),
         tscl::Log::Trace);
  logger("Evaluation set size: " +
                 std::to_string(local_training_collection.getEvaluationSet().getSize()),
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


  logger("Creating scheduler", tscl::Log::Debug);

  ParallelScheduler::Builder scheduler_builder;
  scheduler_builder.setTrainingSets(local_training_collection.getTrainingSet().getTensors(),
                                    local_training_collection.getTargets());
  // Set the resources for the scheduler
  scheduler_builder.setMaxThread(kMaxThread, kAllowMultipleThreadPerDevice);
  scheduler_builder.setDevices(utils::cl_wrapper.getDevices());

  scheduler_builder.setOptimizer(*optimizer);

  // Set the batch size, and allow/disallow batch optimization
  scheduler_builder.setBatchSize(kBatchSize, kAllowBatchDefragmentation);

  auto scheduler = scheduler_builder.build();
  // SchedulerProfiler sc_profiler(scheduler_builder.build(), output_path / "scheduler");
  //  sc_profiler.setVerbose(false);

  ModelEvolutionTracker evaluator(output_path / ("model_evolution_" + std::to_string(rank)), *model,
                                  local_training_collection);

  sleep(rank);
  local_training_collection.display();
  MPI_Barrier(MPI_COMM_WORLD);
  sleep(2);

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
  initializeMPI();
  auto ret = createAndTrain(args[1], args.size() == 3 ? args[2] : "runs/test");
  MPI_Finalize();
  return ret;
}
