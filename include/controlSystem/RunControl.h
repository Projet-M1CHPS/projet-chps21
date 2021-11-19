#pragma once
#include "Image.hpp"
#include "NeuralNetwork.hpp"
#include "NeuralNetworkSerializer.hpp"
#include "Transform.hpp"
#include "controlSystem/RunConfiguration.h"
#include <filesystem>
#include <memory>
#include <vector>

namespace control {

  /** @brief Stores a working directory path and provides functions to return
   * files or sub-folders paths
   *
   */
  class WorkingEnvironnement {
  public:
    WorkingEnvironnement() = default;

    /** @brief Creates a working directory layout and returns a WorkingEnvironnement of the new
     * environnement
     *
     * @param working_dir
     * @return
     */
    static WorkingEnvironnement findOrBuildEnvironnement(std::filesystem::path working_dir);

    [[nodiscard]] std::unique_ptr<RunConfiguration> loadConfiguration() const;
    void cleanup(RunConfiguration const &config) const;

    [[nodiscard]] std::filesystem::path getCachePath() const;
    [[nodiscard]] std::filesystem::path getNeuralNetworkPath() const;

  private:

    std::filesystem::path working_dir;
  };

  class RunResult {
  public:
    explicit RunResult(bool succeeded);
    RunResult(bool succeeded, std::string message);

    RunResult(RunResult const &other) = default;
    RunResult &operator=(RunResult const &other) = default;

    RunResult(RunResult &&other) = default;
    RunResult &operator=(RunResult &&other) = default;

    [[nodiscard]] bool get() const;
    explicit operator bool() const;
    [[nodiscard]] std::string const &getMessage() const;

  private:
    std::string message;
    bool succeeded;
  };

  RunResult runOnConfig(RunConfiguration const &config);

}   // namespace control