#include "controlSystem/RunControl.h"

#include <future>
#include <utility>


namespace fs = std::filesystem;
using namespace image;
using namespace image::transform;
using namespace nnet;

namespace control {

  namespace {

    std::unique_ptr<TrainingImageCache> initCache(WorkingEnvironnement const &env,
                                                  RunConfiguration const &config) {
      auto res = std::make_unique<TrainingImageStash>(env.getCachePath(), config.getInputPath(),
                                                      false);

      return res;
    }

    std::unique_ptr<NeuralNetworkBase> loadNN(WorkingEnvironnement const &env,
                                              RunConfiguration const &config) {
      auto nnet_path = env.getNeuralNetworkPath();
      if (fs::exists(nnet_path)) return NeuralNetworkSerializer::loadFromFile(nnet_path);
      return nullptr;
    }

    std::unique_ptr<NeuralNetworkBase> loadOrCreateNN(WorkingEnvironnement const &env,
                                                      RunConfiguration const &config) {
      std::unique_ptr<NeuralNetworkBase> res;

      if (config.getRunFlags() & config.reuse_network) {
        res = loadNN(env, config);
        if (res) return res;
      }

      res = makeNeuralNetwork(config.getFPPrecision());

      // res->setLayersSize(config.getTopology());

      // res->setActivationFunction(config.getDefaultFunction());
      // res->setActivationFunction(config.getActivationFunctions());
      return res;
    }

  }   // namespace

  std::unique_ptr<WorkingEnvironnement>
  WorkingEnvironnement::findOrBuildEnvironnement(std::filesystem::path working_dir) {
    if (not fs::exists(working_dir)) fs::create_directories(working_dir);

    fs::path tmp = working_dir;
    tmp.append("output");
    if (not fs::exists(tmp)) fs::create_directory(tmp);

    tmp = working_dir;
    tmp.append("cache");
    if (not fs::exists(tmp)) fs::create_directory(tmp);

    return std::make_unique<WorkingEnvironnement>(std::move(working_dir));
  }

  [[nodiscard]] std::unique_ptr<RunConfiguration> WorkingEnvironnement::loadConfiguration() const {
    std::cerr << "FIXME: WorkEnv::loadConf() not implemented" << std::endl;
    return nullptr;
  }

  void WorkingEnvironnement::cleanup(RunConfiguration const &config) const {
    auto flags = config.getRunFlags();

    if (not(flags & RunConfiguration::keep_cache)) { fs::remove_all(getCachePath()); }

    if (flags & RunConfiguration::save_network) {
      std::cerr << "FIXME: Network save not implemented" << std::endl;
    }
  }

  [[nodiscard]] std::filesystem::path WorkingEnvironnement::getCachePath() const {
    auto res = working_dir;
    res.append("cache");
    return res;
  }
  [[nodiscard]] std::filesystem::path WorkingEnvironnement::getNeuralNetworkPath() const {
    auto res = working_dir;
    res.append("NeuralNetwork.nnet");
    return res;
  }

  RunResult AbstractRunController::launch(const RunConfiguration &config) {
    if (not fs::exists(config.getInputPath())) return {false, "Invalid input path"};

    try {
      if (not state or not state->isValid()) setupState(config);

      if (not state or not state->isValid()) throw std::runtime_error("State setup failed");

      run();

    } catch (std::runtime_error &e) { return {false, e.what()}; } catch (...) {
      return {false, "Unknown exception thrown during run."};
    }
    return RunResult(true);
  }

  void TrainingRunController::setupState(const RunConfiguration &config) {
    if (not state) state = std::make_unique<RunState>();

    state->configuration = &config;
    if (not state->environnement) {
      state->environnement =
              WorkingEnvironnement::findOrBuildEnvironnement(config.getTargetDirectory());
    }

    auto &env = *state->environnement;

    if (not state->cache) state->cache = initCache(env, config);
    if (not state->network) state->network = loadOrCreateNN(env, config);
  }

  void TrainingRunController::run() {
    auto &config = *state->configuration;
    state->cache->warmup();

    switch (config.getFPPrecision()) {
      case FloatingPrecision::float32:
        runImpl<float>();
        break;
      case FloatingPrecision::float64:
        runImpl<double>();
        break;
      default:
        throw std::runtime_error("Unknown fp precision");
    }
  }

  void TrainingRunController::cleanup() {
    if (not state) return;

    state->environnement->cleanup(*state->configuration);

    state = nullptr;
  }

  template<typename real>
  void TrainingRunController::runImpl() {
    static_assert(std::is_floating_point_v<real>,
                  "TrainingRunController::runImpl: wrong fp type for run controller");
  }

}   // namespace control