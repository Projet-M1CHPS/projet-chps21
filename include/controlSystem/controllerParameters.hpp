#pragma once

#include "Network.hpp"
#include <filesystem>
#include <memory>
#include <utility>

namespace control {

  class ControllerParameters {
    friend std::ostream &operator<<(std::ostream &os, const ControllerParameters &cp);

  public:
    ControllerParameters(const std::filesystem::path &input_path,
                         const std::filesystem::path &output_path, bool verbose = false);

    std::filesystem::path getOutputPath() { return output_path; }
    void setOutputPath(std::filesystem::path path) { output_path = std::move(path); }

    bool isVerbose() const { return verbose; }
    void setVerbose(bool v) { verbose = v; }

  protected:
    virtual void print(std::ostream &os);

    std::filesystem::path output_path;
    std::filesystem::path input_path;
    bool verbose;
  };

  class TrainingControllerParameters : public ControllerParameters {
  public:
    TrainingControllerParameters(const std::filesystem::path &input_path,
                                 const std::filesystem::path &output_path, size_t max_epoch,
                                 size_t batch_size, bool verbose = false);

    size_t getMaxEpoch() const { return max_epoch; }
    void setMaxEpoch(size_t e) { max_epoch = e; }

    size_t getBatchSize() const { return max_epoch; }
    void setBatchSize(size_t e) { max_epoch = e; }

  private:
    virtual void print(std::ostream &os);

    size_t max_epoch;

    /*Size of the training batch, not to be confused with the size of the batch
                         used during gradient descent*/
    size_t batch_size;
  };

}   // namespace control
