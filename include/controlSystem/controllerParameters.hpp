#pragma once

#include "NeuralNetwork.hpp"
#include "inputSetLoader.hpp"
#include <filesystem>
#include <memory>
#include <utility>

namespace control {

  enum class RunPolicy {
    /** Create a new network and overwrite any previous data
     *
     */
    create = 0,
    /** Try to load an existing network, or create a new one in the failing case. Fails if there's
     * an existing network that doesn't fit the topology or transformations list
     *
     */
    tryLoad,
    /** Load a network, and stop on failure
     *
     */
    load,
    /** Try to load an existing network, creates a new one in the failing case. Overwrite existing
     * network if it doesn't fit the topology or transformations list
     *
     */
    loadOrOverwrite
  };

  /** Abstract generic parameters for a run controller
   *
   */
  class controllerParameters {
  public:
    /** Setup a working environnement and default policies
     *
     * @param input_path
     * @param working_dir_path
     * @param output_path
     */
    explicit controllerParameters(std::filesystem::path input_path)
        : policy(RunPolicy::load), input_path(std::move(input_path)) {}

    // Used to make the class abstract
    virtual ~controllerParameters() = 0;

    /** Returns the path used by the input loader
     *
     * @return
     */
    [[nodiscard]] std::filesystem::path const &getInputPath() const { return input_path; }

    /** Refer to @getInputPath
     *
     * @param path
     */
    void setInputPath(std::filesystem::path path) { input_path = std::move(path); }

    /** Returns the current policy on how previous runs outputs should be handled
     *
     * @return
     */
    [[nodiscard]] RunPolicy getRunPolicy() const { return policy; }

  protected:
    RunPolicy policy;
    std::filesystem::path input_path;
  };

  /** Regroups parameters required for launching a training run
   *
   */
  template<class SetLoader>
  class TrainingParameters : public controllerParameters {
  public:
    /** Setup a working environnement for training a network, along the run policy and the set
     * loader
     *
     * The policy parameter is especially important, as it defines how we should handle previous
     * work encountered
     *
     * @param policy
     * @param input_path
     * @param loader Loader used for loading the training set
     * @param working_path Parent directory of the output_path
     * @param output_path Where the neural network will be save / retrieved, along other data
     */
    TrainingParameters(RunPolicy policy, std::filesystem::path const &input_path,
                       std::shared_ptr<SetLoader> loader, std::filesystem::path working_path,
                       const std::filesystem::path &output_path = "")
        : controllerParameters(input_path), training_method(nnet::TrainingMethod::standard),
          ts_loader(std::move(loader)), working_path(std::move(working_path)) {
      this->policy = policy;
      if (output_path != "") this->output_path = output_path;
      else
        this->output_path = working_path / "output";
    }

    ~TrainingParameters() override = default;

    [[nodiscard]] std::filesystem::path const &getWorkingPath() const { return working_path; }
    void setWorkingPath(std::filesystem::path path) { working_path = std::move(path); }

    [[nodiscard]] SetLoader &getSetLoader() {
      if (not ts_loader)
        throw std::runtime_error("TrainingParameters: Training set loader undefined");
      return *ts_loader;
    }

    [[nodiscard]] SetLoader const &getSetLoader() const {
      if (not ts_loader)
        throw std::runtime_error("TrainingParameters: Training set loader undefined");
      return *ts_loader;
    }

    void setTrainingSetLoader(std::shared_ptr<SetLoader> loader) { ts_loader = std::move(loader); }

    template<class loader, typename... Types>
    void setTrainingSetLoader(Types &&...args) {
      ts_loader = std::make_shared<loader>(std::forward<Types>(args)...);
    }

    [[nodiscard]] nnet::TrainingMethod getTrainingMethod() const { return training_method; }
    void setTrainingMethod(nnet::TrainingMethod tm) { training_method = tm; }

    /** Returns the topology of the network. First element is the input size, and last element is
     * the output. Any element in-between are considered hidden layers.
     *
     * If RunPolicy is set to tryLoad, or loadOrOverwrite, the existing network topology will be
     * checked against this one for equality
     *
     * @return The required topology of the network
     */
    [[nodiscard]] std::vector<size_t> const &getTopology() const { return topology; }

    /** Refer to @getTopology
     *
     * @tparam iterator
     * @param begin
     * @param end
     */
    template<typename iterator>
    void setTopology(iterator begin, iterator end) {
      topology.clear();

      std::copy(begin, end, std::back_inserter(topology));
    }

    /** The how the controller will behave when an existing network is found inside the output_path
     *
     * @param policy
     */
    void setRunPolicy(RunPolicy policy) { this->policy = policy; }


  private:
    nnet::TrainingMethod training_method;
    std::vector<size_t> topology;

    std::shared_ptr<SetLoader> ts_loader;
    std::filesystem::path working_path;
    std::filesystem::path output_path;
  };


  template<class Loader>
  class RunParameters : public controllerParameters {
  public:
    RunParameters(std::filesystem::path const &input_path, std::shared_ptr<Loader> loader,
                  std::filesystem::path network_path, std::filesystem::path output_path = "")
        : controllerParameters(input_path), loader(std::move(loader)),
          network_path(std::move(network_path)), output_path(std::move(output_path)) {}

    [[nodiscard]] std::filesystem::path const &getNetworkPath() const { return network_path; }

    void setNetworkPath(std::filesystem::path path) { network_path = std::move(path); }

    [[nodiscard]] std::filesystem::path const &getOutputPath() const {
      if (output_path == "") return network_path;
      return output_path;
    }

    void setOutputPath(std::filesystem::path path) { output_path = std::move(path); }

    Loader &getSetLoader() {
      if (not loader) throw std::runtime_error("RunParameters: Training set loader undefined");
      return *loader;
    }

    [[nodiscard]] Loader const &getSetLoader() const {
      if (not loader) throw std::runtime_error("RunParameters: Training set loader undefined");
      return *loader;
    }

    void setSetLoader(std::shared_ptr<Loader> new_loader) { loader = std::move(new_loader); }

    template<class sloader, typename... Types>
    void setSetloader(Types &&...args) {
      loader = std::make_shared<sloader>(std::forward<Types>(args)...);
    }


  private:
    std::filesystem::path network_path;
    std::filesystem::path output_path;
    std::shared_ptr<Loader> loader;
  };
}   // namespace control
