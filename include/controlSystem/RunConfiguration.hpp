#pragma once

#include "Image.hpp"
#include "NeuralNetwork.hpp"
#include "NeuralNetworkSerializer.hpp"
#include "Transform.hpp"
#include <filesystem>
#include <memory>
#include <vector>


namespace control {

  /** @brief A class that gathers necessary data to correctly build and start a neural network run
   * Support both training and predicting mode
   *
   * This class takes an input path and a working directory to start a neural network run.
   * Additionally, it should be provided the network topology and a list of transformations to apply
   * on the inputs.
   *
   */
  class RunConfiguration {
  public:
    enum Flags : unsigned { reuse_network = 1, save_network = 2, reuse_cache = 4, keep_cache = 8 };
    enum Mode : unsigned { trainingMode = 0, predictMode };

    RunConfiguration();

    /** @brief Create a default configuration with an an input path and a working directory
     *
     * @param input_path
     * @param working_dir
     */
    RunConfiguration(std::filesystem::path input_path, std::filesystem::path working_dir,
                     std::filesystem::path const &target_dir);

    bool operator==(RunConfiguration const &other) const;

    [[nodiscard]] std::filesystem::path const &getWorkingDirectory() const;
    [[nodiscard]] std::filesystem::path const &getTargetDirectory() const;
    [[nodiscard]] std::filesystem::path const &getInputPath() const;

    [[nodiscard]] unsigned getRunFlags() const { return run_flags; }

    [[nodiscard]] unsigned getCacheFlags() const;
    [[nodiscard]] size_t getCacheSize() const;

    [[nodiscard]] std::vector<image::transform::TransformType> const &getTransformations() const;

    [[nodiscard]] std::vector<af::ActivationFunctionType> const &getActivationFunctions() const;

    /** @brief Return the precision used for the neural network
     *
     * @return
     */
    [[nodiscard]] nnet::FloatingPrecision getFPPrecision() const;

    [[nodiscard]] std::vector<size_t> const &getTopology() const;

    [[nodiscard]] unsigned getMode() const { return mode; }

  private:
    std::filesystem::path input_path, working_dir, target_dir;
    unsigned run_flags;
    Mode mode;

    unsigned cache_flags;
    size_t cache_size;

    std::vector<image::transform::TransformType> transformations;

    std::vector<size_t> topology;
    std::vector<af::ActivationFunctionType> activation_functions;
    nnet::FloatingPrecision precision;
  };

}   // namespace control
