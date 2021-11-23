#pragma once
#include "Image.hpp"
#include "NeuralNetwork.hpp"
#include "NeuralNetworkSerializer.hpp"
#include "Transform.hpp"
#include "controlSystem/ImageCache.hpp"
#include "controlSystem/RunConfiguration.hpp"
#include <filesystem>
#include <memory>
#include <utility>
#include <vector>

namespace control {

  class RunResult {
  public:
    explicit RunResult(bool succeeded) : succeeded(succeeded) {}

    RunResult(bool succeeded, std::string message)
        : succeeded(succeeded), message(std::move(message)) {}

    [[nodiscard]] bool get() const { return succeeded; }

    explicit operator bool() const { return succeeded; }

    [[nodiscard]] std::string const &getMessage() const { return message; }

  private:
    std::string message;
    bool succeeded;
  };

  /** @brief Stores a working directory path and provides functions to return
   * files or sub-folders paths
   *
   */
  class WorkingEnvironnement {
  public:
    WorkingEnvironnement() = default;
    explicit WorkingEnvironnement(std::filesystem::path working_dir)
        : working_dir(std::move(working_dir)) {}

    /** @brief Creates a working directory layout and returns a WorkingEnvironnement of the new
     * environnement
     *
     * @param working_dir
     * @return
     */
    static std::unique_ptr<WorkingEnvironnement>
    findOrBuildEnvironnement(std::filesystem::path working_dir);

    [[nodiscard]] std::unique_ptr<RunConfiguration> loadConfiguration() const;
    void cleanup(RunConfiguration const &config) const;
    [[nodiscard]] std::filesystem::path getCachePath() const;
    [[nodiscard]] std::filesystem::path getNeuralNetworkPath() const;

  private:
    std::filesystem::path working_dir;
  };

  struct RunState {
    std::unique_ptr<nnet::NeuralNetworkBase> network;
    std::unique_ptr<AbstractTrainingCache> cache;
    std::unique_ptr<WorkingEnvironnement> environnement;
    RunConfiguration const *configuration;

    [[nodiscard]] bool isValid() const {
      return network and cache and environnement and configuration;
    }

    void clear() {
      network = nullptr;
      cache = nullptr;
      environnement = nullptr;
      configuration = nullptr;
    }
  };

  class AbstractRunController {
  public:
    AbstractRunController() = default;

    virtual RunResult launch(RunConfiguration const &config);
    virtual ~AbstractRunController() = default;

    AbstractRunController(AbstractRunController const &other) = delete;
    AbstractRunController &operator=(AbstractRunController const &other) = delete;

    AbstractRunController(AbstractRunController &&other) = delete;
    AbstractRunController &operator=(AbstractRunController &&other) = delete;

    [[nodiscard]] RunState *getState() { return state.get(); }

    [[nodiscard]] std::unique_ptr<RunState> yieldState() { return std::move(state); }

    void setState(std::unique_ptr<RunState> other_state) { state = std::move(other_state); }
    virtual void cleanup() = 0;


  protected:
    std::unique_ptr<RunState> state;

  private:
    virtual void run() = 0;
    virtual void setupState(RunConfiguration const &config) = 0;
  };

  class TrainingRunController final : public AbstractRunController {
  public:
    TrainingRunController() = default;
    ~TrainingRunController() override { cleanup(); }

    void cleanup() override;

  private:
    void run() override;

    template<typename real>
    void runImpl();

    void setupState(RunConfiguration const &config) override;
  };

}   // namespace control