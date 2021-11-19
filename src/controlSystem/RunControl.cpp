#include "controlSystem/RunControl.h"
#include "controlSystem/ImageCache.h"
#include "controlSystem/RunConfiguration.h"

#include <future>
#include <utility>


namespace fs = std::filesystem;
using namespace image;
using namespace image::transform;
using namespace nnet;

namespace control {

  using FutureSystem = std::pair<std::future<std::unique_ptr<ImageCache>>,
                                 std::future<std::unique_ptr<NeuralNetworkBase>>>;

  namespace {

    std::unique_ptr<ImageCache> initCache(WorkingEnvironnement const &env,
                                          RunConfiguration const &config) {
      //auto res = std::make_unique<ImageStash>(config.getInputPath());

      //res->warmup();
      return nullptr;
    }

    std::unique_ptr<NeuralNetworkBase> initNeuralNetwork(WorkingEnvironnement const &env,
                                                         RunConfiguration const &config) {
      auto nnet_path = env.getNeuralNetworkPath();
      std::unique_ptr<NeuralNetworkBase> res;

      if (fs::exists(nnet_path)) {
        res = NeuralNetworkSerializer::loadFromFile(nnet_path);
        if (res) return res;
      }

      res = makeNeuralNetwork(config.getFPPrecision());

      // res->setLayersSize(config.getTopology());

      // res->setActivationFunction(config.getDefaultFunction());
      // res->setActivationFunction(config.getActivationFunctions());
      return res;
    }

    FutureSystem initSystem(WorkingEnvironnement const &env, RunConfiguration const &config) {
      return {std::async(std::launch::async, initCache, env, config),
              std::async(std::launch::async, initNeuralNetwork, env, config)};
    }

    void trainNetwork(std::pair<std::future<std::unique_ptr<ImageCache>>,
                                std::future<std::unique_ptr<NeuralNetworkBase>>> &system,
                      RunConfiguration const &config, WorkingEnvironnement const &env) {
      std::cerr << "FIXME: trainNetwork not implemented" << std::endl;
    }

    void runNetwork(FutureSystem &system, RunConfiguration const &config,
                    WorkingEnvironnement const &env) {
      std::cerr << "FIXME: runNetwork not implemented" << std::endl;
    }
  }   // namespace

  WorkingEnvironnement
  WorkingEnvironnement::findOrBuildEnvironnement(std::filesystem::path working_dir) {
    if (not fs::exists(working_dir)) fs::create_directories(working_dir);

    fs::path tmp = working_dir;
    tmp.append("output");
    if (not fs::exists(tmp)) fs::create_directory(tmp);

    tmp = working_dir;
    tmp.append("cache");
    if (not fs::exists(tmp)) fs::create_directory(tmp);

    WorkingEnvironnement res;
    res.working_dir = std::move(working_dir);

    return res;
  }

  std::unique_ptr<RunConfiguration> WorkingEnvironnement::loadConfiguration() const {
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

  std::filesystem::path WorkingEnvironnement::getCachePath() const {
    auto res = working_dir;
    res.append("cache");
    return res;
  }

  std::filesystem::path WorkingEnvironnement::getNeuralNetworkPath() const {
    auto res = working_dir;
    res.append("NeuralNetwork.nnet");
    return res;
  }

  RunResult::RunResult(bool succeeded) : succeeded(succeeded) {}

  RunResult::RunResult(bool succeeded, std::string message)
      : succeeded(succeeded), message(std::move(message)) {}

  bool RunResult::get() const { return succeeded; }

  RunResult::operator bool() const { return succeeded; }

  std::string const &RunResult::getMessage() const { return message; }

  template<typename real>
  void runInit(FutureSystem& system, RunConfiguration const& config) {
    static_assert(std::is_floating_point_v<real>);

    std::unique_ptr<NeuralNetworkBase> base_network = system.second.get();
    auto network = dynamic_cast<NeuralNetwork<real> *>(base_network.get());

    std::unique_ptr<ImageCache> cache = system.first.get();

    if (not cache or not network)
      throw std::runtime_error("Invalid ptr during run startup");

    switch (config.getMode()) {
      case RunConfiguration::trainingMode :
        //launchTraining<real>(*network, *cache);
        break;

      case RunConfiguration::predictMode :
        //launchPredict<real>(*network, *cache);
        break;
    }
  }

  RunResult runOnConfig(RunConfiguration const &config) {
    if (not fs::exists(config.getInputPath())) return {false, "Invalid input path"};

    try {
      auto &wd = config.getTargetDirectory();

      WorkingEnvironnement env = WorkingEnvironnement::findOrBuildEnvironnement(wd);

      FutureSystem system = initSystem(env, config);

      switch (config.getFPPrecision()) {
        case FloatingPrecision::float32:
          runInit<float>(system, config);
          break;
        case FloatingPrecision::float64:
          runInit<double>(system, config);
          break;
        default:
          throw std::runtime_error("Unknown fp precision");
      }
      env.cleanup(config);

    } catch (std::runtime_error &e) { return {false, e.what()}; } catch (...) {
      return {false, "Unknown exception thrown during run."};
    }
    return RunResult(true);
  }
}   // namespace control