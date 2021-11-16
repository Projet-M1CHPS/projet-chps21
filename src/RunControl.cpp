#include "RunControl.h"

#include <search.h>
#include <utility>


namespace fs = std::filesystem;
using namespace image;
using namespace image::transform;
using namespace nnet;

namespace {

  WorkingEnvironnement findOrCreateWorkingEnv(fs::path const &working_dir,
                                              RunConfiguration const &config) {
    if (working_dir != "" and not fs::exists(working_dir)) fs::create_directories(working_dir);

    /*WorkingEnvironnement res;
    for (auto const &dir : fs::directory_iterator(working_dir)) {
      auto curr = WorkingEnvironnement::make(dir);

      if (config == curr.loadConfiguration()) {
        return res;
      }
    }
*/
    return WorkingEnvironnement::make(working_dir);
  }

  std::unique_ptr<ImageCache> initCache(WorkingEnvironnement const &env,
                                        RunConfiguration const &config) {
    /* auto res = std::make_unique<ImageCache>(config.getInputPath(), config.getCacheSize());

     res->setCacheDirectory(env.getCachePath());
     res->setTransformEngine(TransformEngine::make(config.getTransformations()));
     res->setFlags(config.getCacheFlags());

     res->warmup();*/
    return std::make_unique<ImageCache>();
  }

  std::unique_ptr<NeuralNetworkBase> initNeuralNetwork(WorkingEnvironnement const &env,
                                                       RunConfiguration const &config) {
    auto nnet_path = env.getNeuralNetworkPath();
    std::unique_ptr<NeuralNetworkBase> res;

    if (fs::exists(nnet_path)) res = NeuralNetworkSerializer::loadFromFile(nnet_path);

    if (res) return res;

    switch (config.getFPPrecision()) {
      case FloatingPrecision::float32:
        res = std::make_unique<NeuralNetwork<float>>();
        break;
      case FloatingPrecision::float64:
        res = std::make_unique<NeuralNetwork<double>>();
        break;
    }

    res->setLayersSize(config.getTopology());
    res->setActivationFunction(af::ActivationFunctionType::leakyRelu);
    // res->setActivationFunction(config.getActivationFunctions());
    res->randomizeSynapses();
    return res;
  }

  void trainNetwork(NeuralNetworkBase &network, ImageCache &cache, RunConfiguration const &config,
                    WorkingEnvironnement const &env) {
    std::cerr << "FIXME: trainNetwork not implemented" << std::endl;
  }

  void runNetwork(NeuralNetworkBase &network, ImageCache &cache, RunConfiguration const &config,
                  WorkingEnvironnement const &env) {
    std::cerr << "FIXME: runNetwork not implemented" << std::endl;
  }
}   // namespace

RunConfiguration::RunConfiguration()
    : flags(Flags::save_network | Flags::reuse_network | Flags::reuse_cache | Flags::keep_cache),
      cache_size(16_mb), cache_flags(ImageCache::deferred),
      precision(nnet::FloatingPrecision::float32), working_dir("series"), mode(runMode) {}

RunConfiguration::RunConfiguration(std::filesystem::path input_path,
                                   std::filesystem::path working_dir)
    : RunConfiguration() {
  this->input_path = std::move(input_path);
  this->working_dir = std::move(working_dir);
}


bool RunConfiguration::operator==(RunConfiguration const &other) const { return false; }

std::filesystem::path const &RunConfiguration::getWorkingDirectory() const { return working_dir; }

std::filesystem::path const &RunConfiguration::getInputPath() const { return input_path; }

unsigned RunConfiguration::getCacheFlags() const { return cache_flags; }

size_t RunConfiguration::getCacheSize() const { return cache_size; }

std::vector<image::transform::TransformType> const &RunConfiguration::getTransformations() const {
  return transformations;
}

std::vector<af::ActivationFunctionType> const &RunConfiguration::getActivationFunctions() const {
  return activation_functions;
}

nnet::FloatingPrecision RunConfiguration::getFPPrecision() const { return precision; }

std::vector<size_t> const &RunConfiguration::getTopology() const { return topology; }

WorkingEnvironnement WorkingEnvironnement::make(std::filesystem::path working_dir) {
  return WorkingEnvironnement(std::move(working_dir));
}

RunConfiguration WorkingEnvironnement::loadConfiguration() const {
  std::cerr << "FIXME: WorkEnv::loadConf() not implemented" << std::endl;
  return {};
}

void WorkingEnvironnement::cleanup(RunConfiguration const &config) const {
  std::cerr << "FIXME: WorkEnv::cleanup() not implemented" << std::endl;
}

std::filesystem::path WorkingEnvironnement::getCachePath() const {
  auto res = working_dir;
  res.append("/cache");
  return res;
}

std::filesystem::path WorkingEnvironnement::getNeuralNetworkPath() const {
  auto res = working_dir;
  res.append("/NeuralNetwork.nnet");
  return res;
}

WorkingEnvironnement::WorkingEnvironnement(std::filesystem::path working_dir)
    : working_dir(std::move(working_dir)) {

  if (not fs::exists(working_dir))
    fs::create_directories(working_dir);

  fs::path tmp = working_dir;
  tmp.append("/output");
  if (not fs::exists(tmp))
    fs::create_directory(tmp);

  tmp = working_dir;
  tmp.append("/cache");
  if (not fs::exists(tmp))
    fs::create_directory(tmp);
}

bool runOnConfig(RunConfiguration const &config) {
  if (not fs::exists(config.getInputPath())) return false;

  try {
    auto &wd = config.getWorkingDirectory();

    WorkingEnvironnement env = findOrCreateWorkingEnv(wd, config);

    std::unique_ptr<ImageCache> cache = initCache(env, config);
    std::unique_ptr<NeuralNetworkBase> network = initNeuralNetwork(env, config);

    if (config.getMode() == RunConfiguration::trainMode)
      trainNetwork(*network, *cache, config, env);
    else if (config.getMode() == RunConfiguration::runMode)
      runNetwork(*network, *cache, config, env);
    env.cleanup(config);
  } catch (std::runtime_error &e) {
    std::cerr << "Runtime exception thrown during run: " << e.what() << std::endl;
    return false;
  } catch (...) {
    std::cerr << "Unknown exception thrown during run." << std::endl;
    return false;
  }
  return true;
}
