#include "controlSystem/RunControl.hpp"
#include <controlSystem/StatTracker.hpp>

#include <fstream>
#include <future>
#include <utility>


namespace fs = std::filesystem;
using namespace image;
using namespace image::transform;
using namespace nnet;

namespace control {

  namespace {

    /** @brief Create a TCache including its initialization from an input path
     *
     * @tparam TCache
     * @param input
     * @param shuffle_input
     * @return
     */
    template<typename TCache>
    std::unique_ptr<AbstractImageCache> createCache(std::filesystem::path const &input,
                                                    bool shuffle_input, bool verbose) {
      auto res = std::make_unique<TCache>(input, shuffle_input);
      res->setTargetSize(16, 16);
      res->init(verbose, std::cout);

      if (not res) throw std::runtime_error("initCache(): Cache creation failed");
      return res;
    }

    std::unique_ptr<NeuralNetworkBase> createNN(std::vector<size_t> const &topology,
                                                FloatingPrecision precision) {
      std::unique_ptr<NeuralNetworkBase> res = makeNeuralNetwork(precision);

      size_t nlayer = topology.size();
      res->setLayersSize(topology);
      res->setActivationFunction(af::ActivationFunctionType::leakyRelu);

      // FIXME: to prevent leaky relu explosion, we alternate leaky relu with sigmoid
      // We also want the last layer to be a sigmoid
      for (size_t i = 0; i < nlayer; i++) {
        if (i == nlayer - 2 or i % 3 == 0)
          res->setActivationFunction(af::ActivationFunctionType::sigmoid, i);
      }
      res->randomizeSynapses();
      return res;
    }

  }   // namespace

  std::unique_ptr<WorkingEnvironnement>
  WorkingEnvironnement::findOrBuildEnvironnement(std::filesystem::path working_dir) {
    if (not fs::exists(working_dir)) fs::create_directories(working_dir);

    fs::path tmp = working_dir / "output";
    if (not fs::exists(tmp)) fs::create_directory(tmp);

    return std::make_unique<WorkingEnvironnement>(std::move(working_dir));
  }

  [[nodiscard]] std::unique_ptr<RunConfiguration> WorkingEnvironnement::loadConfiguration() {
    std::cerr << "FIXME: WorkEnv::loadConf() not implemented" << std::endl;
    return nullptr;
  }

  RunResult AbstractRunController::launch(const RunConfiguration &config) {
    if (not fs::exists(config.getInputPath()))
      return {false, "AbstractRunController: Invalid input path"};

    try {
      if (not state or not state->isValid()) setupState(config);
      run();

    } catch (std::exception &e) { return {false, e.what()}; }
    return RunResult(true);
  }

  void TrainingRunController::setupState(const RunConfiguration &config) {
    if (not state) state = std::make_unique<ControllerState>();

    state->configuration = &config;
    if (not state->environnement) {
      state->environnement =
              WorkingEnvironnement::findOrBuildEnvironnement(config.getTargetDirectory());
      if (config.isVerbose())
        config.out() << "Setup working directory " << config.getTargetDirectory() << std::endl;
    }

    auto &env = *state->environnement;

    if (not state->cache) {
      if (config.isVerbose())
        config.out() << "Setup cache with input directory: " << config.getInputPath() << std::endl;
      switch (config.getFPPrecision()) {
        case FloatingPrecision::float32:
          state->cache = createCache<TrainingStash<float>>(config.getInputPath(), true,
                                                           config.isVerbose());
          break;
        case FloatingPrecision::float64:
          state->cache = createCache<TrainingStash<double>>(config.getInputPath(), true,
                                                            config.isVerbose());
          break;
      }
    }
    if (not state->network) {
      // FIXME: placeholder topology
      std::vector<size_t> topology = {16 * 16, 64, 32, 32, 8, 2};
      state->network = createNN(topology, config.getFPPrecision());
    }
  }

  void TrainingRunController::run() {
    auto precision = state->configuration->getFPPrecision();

    switch (precision) {
      case FloatingPrecision::float32:
        runImpl<float>();
        break;
      case FloatingPrecision::float64:
        runImpl<double>();
        break;
      default:
        throw std::runtime_error("TrainingRunController: Unknown fp precision");
    }
  }

  void TrainingRunController::cleanup() {
    if (not state) return;

    state = nullptr;
  }

  // Main method where the training occurs
  template<typename real>
  void TrainingRunController::runImpl() {
    static_assert(std::is_floating_point_v<real>,
                  "TrainingRunController::runImpl: wrong fp type for run controller");

    auto &config = *state->configuration;
    auto &cache = dynamic_cast<AbstractTrainingCache<real> &>(*state->cache);
    auto &nn = dynamic_cast<NeuralNetwork<real> &>(*state->network);

    real initial_learning_rate = 1.f, learning_rate = 0.;
    size_t batch_size = 10, max_epoch = 100;
    if (config.isVerbose())
      config.out() << std::setprecision(8)
                   << "Training started with: {initial_learning_rate: " << initial_learning_rate
                   << ", batch_size: " << batch_size << ", Max epoch: " << max_epoch << "}"
                   << std::endl;


    auto const &classes = cache.getClasses();
    CTracker stracker(state->environnement->getOutputPath(), classes.begin(), classes.end());
    math::Matrix<size_t> confusion(classes.size(), classes.size());
    math::Matrix<real> target(classes.size(), 1);

    while (stracker.getEpoch() < max_epoch) {
      learning_rate =
              initial_learning_rate * (1 / (1 + 0.9 * static_cast<double>(stracker.getEpoch())));
      for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < cache.getTrainingSetSize(); j++) {
          auto type = cache.getTrainingType(j);
          target.fill(0);
          target(type, 0) = 1.f;

          nn.train(cache.getTraining(j), target, learning_rate);
        }
      }

      confusion.fill(0);
      for (int i = 0; i < cache.getEvalSetSize(); i++) {
        auto type = cache.getEvalType(i);
        auto res = nn.predict(cache.getEval(i));
        auto res_type = std::distance(res.begin(), std::max_element(res.begin(), res.end()));

        confusion(res_type, type)++;
      }
      auto stats = stracker.computeStats(confusion);
      if (config.isVerbose()) config.out() << stats;
      stats.dumpToFiles();
      stracker.nextEpoch();
    }

    if (not config.isVerbose()) return;

    for (int i = 0; i < cache.getEvalSetSize(); i++) {
      auto type = cache.getEvalType(i);
      auto res = nn.predict(cache.getEval(i));
      auto res_type = std::distance(res.begin(), std::max_element(res.begin(), res.end()));

      config.out() << "[Image: " << i << "]: type " << classes[type] << "(" << type
                   << ") output :\n"
                   << res;

      confusion(res_type, type)++;
    }
    config.out() << std::endl;
    auto stats = stracker.computeStats(confusion);
    stats.classification_report(config.out(), classes);
  }

}   // namespace control