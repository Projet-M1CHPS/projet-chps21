#include <nlohmann/json.hpp>

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
using json = nlohmann::json;


class NetworkInterface {
private:
  std::string parameterFilepath;

  json readJSONConfigFile() const {
    std::ifstream i(parameterFilepath);
    json json_obj;
    i >> json_obj;
    i.close();
    return json_obj;
  }

  void setupLogger() {
    auto &handler = logger.addHandler<StreamLogHandler>("term", std::cout);
    handler.tsType(tscl::timestamp_t::Partial);
    handler.minLvl(Log::Trace);
    // handler.enable(false);

    auto &thandler = logger.addHandler<StreamLogHandler>("file", "logs.txt");
    thandler.minLvl(Log::Information);
  }

public:
  NetworkInterface(std::string parameter_filepath) : parameterFilepath(parameter_filepath){};
  ~NetworkInterface() = default;

  json getJSONConfig() const { return readJSONConfigFile(); }
  void printJSONConfig() const { std::cout << std::setw(4) << getJSONConfig() << std::endl; }

  bool createAndTrain() {
    auto config = getJSONConfig();
    std::filesystem::path input_path = std::string(config["input_path"]);
    std::filesystem::path output_path = std::string(config["output_path"]);

    int resize_dimension = config["training"]["resize"];
    bool use_logger = config["training"]["use_logger"];


    setupLogger();
    tscl::logger("Current version: " + tscl::Version::current.to_string(), tscl::Log::Debug);
    tscl::logger("Fetching model from  " + input_path.string(), tscl::Log::Debug);
    tscl::logger("Output path: " + output_path.string(), tscl::Log::Debug);

    if (not std::filesystem::exists(output_path)) std::filesystem::create_directories(output_path);

    tscl::logger("Creating collection loader", tscl::Log::Debug);
    CITCLoader loader(resize_dimension, resize_dimension);
    auto &pre_engine = loader.getPreProcessEngine();
    // Add preprocessing transformations here
    pre_engine.addTransformation(std::make_shared<image::transform::Inversion>());
    auto &engine = loader.getPostProcessEngine();
    // Add postprocessing transformations here
    engine.addTransformation(std::make_shared<image::transform::BinaryScale>());

    tscl::logger("Loading collection", tscl::Log::Information);
    auto training_collection = loader.load(input_path);

    // Create a correctly-sized topology
    nnet::MLPTopology topology = {32 * 32, 64, 64, 32, 32};
    topology.push_back(training_collection->getClassCount());

    auto model = nnet::MLPModelFactory::randomSigReluAlt(topology);

    auto tm = std::make_shared<nnet::DecayMomentumOptimization>(model->getPerceptron(), 0.2, 0.1,
                                                                0.9);

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
};