#pragma once

#include "Network.hpp"
#include <filesystem>
#include <memory>
#include <utility>

namespace control {

  /** Utility class to store the parameters of the controller.
   *
   */
  class ControllerParameters {
    friend std::ostream &operator<<(std::ostream &os, const ControllerParameters &cp);

  public:
    /**
     *
     * @param input_path Path to the input directory. This can be a single file, or a directory
     * @param output_path Path where any output data will be saved
     * @param verbose Wheter the controller is verbose or not
     */
    ControllerParameters(std::filesystem::path input_path, std::filesystem::path output_path,
                         bool verbose = false);

    /**
     * @return Path where any output data will be saved
     */
    std::filesystem::path getOutputPath() { return output_path; }
    void setOutputPath(std::filesystem::path path) { output_path = std::move(path); }

    /**
     * @return True if the controller must be verbose about what it is doing, false otherwise
     */
    [[nodiscard]] bool isVerbose() const { return verbose; }
    void setVerbose(bool v) { verbose = v; }

  protected:
    virtual void print(std::ostream &os) const;

    std::filesystem::path input_path;
    std::filesystem::path output_path;
    bool verbose;
  };

  /** Extension of ControllerParameters that stores additional data for a training run
   *
   */
  class TrainingControllerParameters : public ControllerParameters {
  public:
    TrainingControllerParameters(const std::filesystem::path &input_path,
                                 const std::filesystem::path &output_path, size_t max_epoch,
                                 size_t batch_size, bool verbose = false);

    /**
     * @return The maximum training epoch at which the controller will stop
     */
    [[nodiscard]] size_t getMaxEpoch() const { return max_epoch; }
    void setMaxEpoch(size_t e) { max_epoch = e; }

    /** This function return the training batch size. This must not be confused with the
     * batch size of the gradient descent for batch or mini-batch optimizers.
     *
     *
     * @return The size of the batch used for training
     */
    [[nodiscard]] size_t getBatchSize() const { return batch_size; }
    void setBatchSize(size_t e) { batch_size = e; }

  protected:
    void print(std::ostream &os) const override;

    size_t max_epoch;

    /*Size of the training batch, not to be confused with the size of the batch
                         used during gradient descent*/
    size_t batch_size;
  };

}   // namespace control
