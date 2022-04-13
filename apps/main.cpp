#include "EvalController.hpp"
#include "MPITrainingController.hpp"
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
  engine.addTransformation(std::make_shared<image::transform::BinaryScale>());

  TrainingCollection training_collection = loader.load(input_path);

  tscl::logger("Training set size: " +
                       std::to_string(training_collection.getTrainingSet().getSize()),
               tscl::Log::Trace);
  tscl::logger("Testing set size: " +
                       std::to_string(training_collection.getEvaluationSet().getSize()),
               tscl::Log::Trace);


  // Create a correctly-sized topology
  nnet::MLPTopology topology = {kImageSize * kImageSize, 64, 64, 64};
  topology.pushBack(training_collection.getClassCount());

  auto model = nnet::MLPModel::randomReluSigmoid(topology);
  // auto model = nnet::MLPModel::random(topology, af::ActivationFunctionType::leakyRelu);
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


bool MPI_createAndTrain(std::filesystem::path const &input_path,
                        std::filesystem::path const &output_path) {
  tscl::logger("Current version: " + tscl::Version::current.to_string(), tscl::Log::Debug);
  tscl::logger("Fetching input from  " + input_path.string(), tscl::Log::Debug);
  tscl::logger("Output path: " + output_path.string(), tscl::Log::Debug);
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  constexpr int kImageSize = 32;
  // Ensure this is the same size as the batch size
  constexpr int kTensorSize = 256;
  // Empty for the children (rank > 0)
  TrainingCollection training_collection(kImageSize, kImageSize);

  // Variables shared between processes with MPI
  int class_count = 0;
  // ===============================


  if (rank == 0) {
    if (not std::filesystem::exists(output_path)) std::filesystem::create_directories(output_path);


    tscl::logger("Loading dataset", tscl::Log::Debug);
    TrainingCollectionLoader loader(kTensorSize, kImageSize, kImageSize);
    auto &pre_engine = loader.getPreProcessEngine();
    // Add preprocessing transformations here
    pre_engine.addTransformation(std::make_shared<image::transform::Inversion>());

    auto &engine = loader.getPostProcessEngine();
    // Add postprocessing transformations here
    engine.addTransformation(std::make_shared<image::transform::BinaryScale>());

    training_collection = loader.load(input_path);

    tscl::logger("Training set size: " +
                         std::to_string(training_collection.getTrainingSet().getSize()),
                 tscl::Log::Trace);
    tscl::logger("Testing set size: " +
                         std::to_string(training_collection.getEvaluationSet().getSize()),
                 tscl::Log::Trace);

    class_count = (int) training_collection.getClassCount();
  }
  MPI_Bcast(&class_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
  tscl::logger("[P" + std::to_string(rank) + "]: Class count: " + std::to_string(class_count),
               tscl::Log::Debug);
  // Create a correctly-sized topology
  nnet::MLPTopology topology = {kImageSize * kImageSize, 64, 64, 32, 16};
  topology.pushBack(class_count);

  auto model = nnet::MLPModel::randomReluSigmoid(topology);

  auto optimizer = nnet::MLPBatchOptimizer::make<nnet::SGDOptimization>(*model, 0.03);

  tscl::logger("[P" + std::to_string(rank) + "]: Creating controller", tscl::Log::Trace);
  // EvalController controller(output_path, model.get(), &training_collection.getEvaluationSet());
  MPITrainingController controller(output_path, *model, *optimizer, training_collection);
  ControllerResult res = controller.run();

  if (rank == 0) {
    if (not res) {
      tscl::logger("[P" + std::to_string(rank) + "]: Controller failed with an exception",
                   tscl::Log::Error);
      tscl::logger(res.getMessage(), tscl::Log::Error);
      return false;
    }
    nnet::MLPModelSerializer::writeToFile(output_path / "model.nnet", *model);
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
  // utils::clWrapper::initOpenCL(*utils::clWrapper::makeDefault());
  utils::clWrapper::initOpenCL(*utils::clWrapper::makeDefault());

  std::vector<std::string> args;
  for (size_t i = 0; i < argc; i++) args.emplace_back(argv[i]);


#ifdef USE_MPI
  #pragma message("MPI included")
  MPI_Init(&argc, &argv);
  auto ret = MPI_createAndTrain(args[1], args.size() == 3 ? args[2] : "runs/test");
  MPI_Finalize();
  return ret;
#else
  #pragma message("MPI not included")
  return createAndTrain(args[1], args.size() == 3 ? args[2] : "runs/test");
  // predict_test();
  // return 0;
#endif

  return 0;
}
