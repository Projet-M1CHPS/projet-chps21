#include "NetworkInterface.hpp"

#pragma region anonymous_utils

bool isJsonSectionEnabled(const json &e) {
  return e.contains("enabled") && e["enabled"].is_boolean() && e["enabled"];
}

std::unique_ptr<nnet::MLPModel> getModelFromConfig(const json &modelsConfig,
                                                   const nnet::MLPTopology &topology) {
  if (isJsonSectionEnabled(modelsConfig["random"])) return nnet::MLPModelFactory::random(topology);
  if (isJsonSectionEnabled(modelsConfig["randomsigrelu"]))
    return nnet::MLPModelFactory::randomSigReluAlt(topology);
  throw std::invalid_argument("No model defined. Should have been detected before. Report this.");
}

std::shared_ptr<nnet::OptimizationMethod>
getOptimizationFromConfig(const json &optimizationConfig,
                          const std::unique_ptr<nnet::MLPModel> &model) {
  if (isJsonSectionEnabled(optimizationConfig["sgd"]))
    return std::make_shared<nnet::SGDOptimization>((float) optimizationConfig["sgd"]["lr"]);

  if (isJsonSectionEnabled(optimizationConfig["decay"]))
    return std::make_shared<nnet::DecayOptimization>(optimizationConfig["decay"]["lr0"],
                                                     optimizationConfig["decay"]["dr"]);
  if (isJsonSectionEnabled(optimizationConfig["momentum"]))
    return std::make_shared<nnet::MomentumOptimization>(model->getPerceptron(),
                                                        optimizationConfig["momentum"]["lr"],
                                                        optimizationConfig["momentum"]["mom"]);
  if (isJsonSectionEnabled(optimizationConfig["decaymomentum"]))
    return std::make_shared<nnet::DecayMomentumOptimization>(
            model->getPerceptron(), optimizationConfig["decaymomentum"]["lr0"],
            optimizationConfig["decaymomentum"]["dr"], optimizationConfig["decaymomentum"]["mom"]);
  if (isJsonSectionEnabled(optimizationConfig["rprop"]))
    return std::make_shared<nnet::RPropPOptimization>(
            model->getPerceptron(), optimizationConfig["rprop"]["eta_p"],
            optimizationConfig["rprop"]["eta_m"], optimizationConfig["rprop"]["lr_max"],
            optimizationConfig["rprop"]["lr_min"]);
  throw std::invalid_argument(
          "No optimization defined. Should have been detected before. Report this.");
}

std::shared_ptr<nnet::MLPBatchOptimizer>
getOptimizerFromConfig(const json &optimizerConfig, const std::unique_ptr<nnet::MLPModel> &model,
                       const std::shared_ptr<nnet::OptimizationMethod> &optimization) {
  if (isJsonSectionEnabled(optimizerConfig["minibatch"]))
    return std::make_unique<nnet::MLPMiniBatchOptimizer>(*model, optimization,
                                                         optimizerConfig["minibatch"]["size"]);

  if (isJsonSectionEnabled(optimizerConfig["batch"]))
    return std::make_unique<nnet::MLPBatchOptimizer>(*model, optimization);

  throw std::invalid_argument(
          "No optimizer defined. Should have been detected before. Report this.");
}

#pragma endregion

#pragma region Signals handling

void signalHandler(int signal_num) {
  switch (signal_num) {
    case SIGINT:
      std::cout << "CPP_INTERFACE: Ignored SIGINT (Ctrl+C), use SIGQUIT / SIGABRT instead for soft "
                   "end."
                << std::endl;
      break;
    case SIGQUIT:
      std::cout << "CPP_INTERFACE: Signal SIGQUIT trapped." << std::endl;
      exit(signal_num);
    case SIGABRT:
      std::cout << "CPP_INTERFACE: Signal SIGABRT trapped." << std::endl;
      exit(signal_num);
    default:
      std::cout << "CPP_INTERFACE: Unhandled signal " << signal_num << ". Report this."
                << std::endl;
  }
}

void initSignalHandler() {
  signal(SIGINT, signalHandler);
  signal(SIGQUIT, signalHandler);
  signal(SIGABRT, signalHandler);
}

#pragma endregion

#pragma region private_section

json NetworkInterface::readJSONConfig(const std::string &config_file_path) {
  std::ifstream i(config_file_path);
  json json_obj;
  i >> json_obj;
  i.close();
  return json_obj;
}

json NetworkInterface::readJSONConfig() const { return readJSONConfig(parameterFilepath); }

json NetworkInterface::getJSONConfig() const {
  json cfg = readJSONConfig();
  // TODO: Remove unnecessary attributes
  checkConfigValid(cfg);
  return cfg;
}

void NetworkInterface::printJSONConfig() const {
  std::cout << std::setw(4) << getJSONConfig() << std::endl;
}


void NetworkInterface::checkConfigValid(const json &config) {
  if (!config.contains("input_path") || !config["input_path"].is_string())
    throw std::invalid_argument("Input_path not/badly defined in the configuration");

  if (!config.contains("output_path") || !config["output_path"].is_string())
    throw std::invalid_argument("Output_path not/badly defined in the configuration");

  if (!config.contains("training") || !config["training"].is_object())
    throw std::invalid_argument("Training configuration section not/badly defined");

  json trainingConfig = config["training"];
  if (!(trainingConfig.contains("resize") && trainingConfig["resize"].is_number_integer()))
    throw std::invalid_argument("Resize not/badly defined in the configuration");

  if (!(trainingConfig.contains("preprocess_transformations") &&
        trainingConfig["preprocess_transformations"].is_array()))
    throw std::invalid_argument(
            "Preprocess_transformations not/badly defined in the configuration");

  if (!(trainingConfig.contains("postprocess_transformations") &&
        trainingConfig["postprocess_transformations"].is_array()))
    throw std::invalid_argument(
            "Postprocess_transformations not/badly defined in the configuration");

  if (!trainingConfig.contains("max_epoch") || !trainingConfig["max_epoch"].is_number())
    throw std::invalid_argument("Max_epoch not/badly defined in the configuration");
  if (!trainingConfig.contains("batch_size") || !trainingConfig["batch_size"].is_number())
    throw std::invalid_argument("Batch_size not/badly defined in the configuration");
  if (!trainingConfig.contains("verbose") || !trainingConfig["verbose"].is_boolean())
    throw std::invalid_argument("Verbose not/badly defined in the configuration");
  if (!trainingConfig.contains("topology") || !trainingConfig["topology"].is_array() ||
      !std::all_of(trainingConfig["topology"].begin(), trainingConfig["topology"].end(),
                   [](const json &e) { return e.is_number(); }))
    throw std::invalid_argument("Topology not/badly defined in the configuration");

  if (!(trainingConfig.contains("activation") && trainingConfig["activation"].is_object()))
    throw std::invalid_argument("Activations configuration section not/badly defined");
  json activations = trainingConfig["activation"];
  if (!std::any_of(activations.begin(), activations.end(), isJsonSectionEnabled))
    throw std::invalid_argument("No activation enabled in the configuration");

  if (!(trainingConfig.contains("optimization") && trainingConfig["optimization"].is_object()))
    throw std::invalid_argument("Optimizations configuration section not/badly defined");
  json optimizations = trainingConfig["activation"];
  if (!std::any_of(optimizations.begin(), optimizations.end(), isJsonSectionEnabled))
    throw std::invalid_argument("No optimization enabled in the configuration");

  if (!(trainingConfig.contains("optimizer") && trainingConfig["optimizer"].is_object()))
    throw std::invalid_argument("Optimizers configuration section not/badly defined");
  json optimizers = trainingConfig["optimizer"];
  if (!std::any_of(optimizers.begin(), optimizers.end(), isJsonSectionEnabled))
    throw std::invalid_argument("No optimizer enabled in the configuration");

  if (!(trainingConfig.contains("model") && trainingConfig["model"].is_object()))
    throw std::invalid_argument("Models configuration section not/badly defined");
  json models = trainingConfig["model"];
  if (!std::any_of(models.begin(), models.end(), isJsonSectionEnabled))
    throw std::invalid_argument("No model enabled in the configuration");
}

void NetworkInterface::setupLogger() {
  auto &handler = logger.addHandler<StreamLogHandler>("term", std::cout);
  handler.tsType(tscl::timestamp_t::Partial);
  handler.minLvl(Log::Trace);
  // handler.enable(false);

  auto &logHandler = logger.addHandler<StreamLogHandler>("file", "logs.txt");
  logHandler.minLvl(Log::Information);
}

#pragma endregion

#pragma region public_section

bool NetworkInterface::createAndTrain() {
  json config = getJSONConfig();
  json trainingConfig = config["training"];
  checkConfigValid(config);
  std::filesystem::path input_path = std::string(config["input_path"]);
  std::filesystem::path output_path = std::string(config["output_path"]);

  initSignalHandler();

  if (trainingConfig["use_logger"]) {
    setupLogger();
    tscl::logger("Current version: " + tscl::Version::current.to_string(), tscl::Log::Debug);
    tscl::logger("Fetching model from  " + input_path.string(), tscl::Log::Debug);
    tscl::logger("Output path: " + output_path.string(), tscl::Log::Debug);
  }

  if (not std::filesystem::exists(input_path)) {
    tscl::logger("Input path is not valid", tscl::Log::Error);
    return false;
  }

  if (not std::filesystem::exists(output_path)) std::filesystem::create_directories(output_path);

  tscl::logger("Creating collection loader", tscl::Log::Debug);
  int resize_dimension = trainingConfig["resize"];
  CITCLoader loader(resize_dimension, resize_dimension);

  tscl::logger("Image processing:", tscl::Log::Debug);
  tscl::logger("\tpre-processing", tscl::Log::Debug);
  auto &pre_engine = loader.getPreProcessEngine();
  // Add preprocessing transformations here
  std::for_each(trainingConfig["preprocess_transformations"].begin(),
                trainingConfig["preprocess_transformations"].end(),
                [&pre_engine](const std::string &e) {
                  auto tr = image::transform::TransformEngine::getTransformationFromString(e);
                  pre_engine.addTransformation(tr);
                });

  tscl::logger("\tpost-processing", tscl::Log::Debug);
  auto &engine = loader.getPostProcessEngine();
  // Add postprocessing transformations here
  std::for_each(trainingConfig["postprocess_transformations"].begin(),
                trainingConfig["postprocess_transformations"].end(),
                [&engine](const std::string &e) {
                  auto tr = image::transform::TransformEngine::getTransformationFromString(e);
                  engine.addTransformation(tr);
                });

  tscl::logger("Loading collection", tscl::Log::Debug);
  auto training_collection = loader.load(input_path);

  tscl::logger("Creating topology", tscl::Log::Debug);
  // Create a correctly-sized topology
  std::vector<size_t> topology_config = trainingConfig["topology"];
  nnet::MLPTopology topology(topology_config);

  topology.push_back(training_collection->getClassCount());

  tscl::logger("Creating model", tscl::Log::Debug);
  auto model = getModelFromConfig(trainingConfig["model"], topology);

  tscl::logger("Creating optimization", tscl::Log::Debug);
  auto optimization = getOptimizationFromConfig(trainingConfig["optimization"], model);

  tscl::logger("Creating optimizer", tscl::Log::Debug);
  auto optimizer = getOptimizerFromConfig(trainingConfig["optimizer"], model, optimization);

  tscl::logger("Creating controller", tscl::Log::Trace);
  TrainingControllerParameters parameters(input_path, output_path, trainingConfig["max_epoch"],
                                          trainingConfig["batch_size"], trainingConfig["verbose"]);
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

void NetworkInterface::onPrecisionChanged(const std::function<void(float)> &callback) const {
  callback(((float) ((float) rand()) / ((float) RAND_MAX)));
}

#pragma endregion
