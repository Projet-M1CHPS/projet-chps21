#pragma once

#include "Network.hpp"
#include <filesystem>
#include <memory>
#include <utility>

namespace control {

  class ControllerParameters {
    friend std::ostream &operator<<(std::ostream &os, const ControllerParameters &cp);

  public:
    ControllerParameters(std::filesystem::path input_path, std::filesystem::path output_path,
                         bool verbose = false);

    std::filesystem::path getOutputPath() { return output_path; }
    void setOutputPath(std::filesystem::path path) { output_path = std::move(path); }

    [[nodiscard]] bool isVerbose() const { return verbose; }
    void setVerbose(bool v) { verbose = v; }

  protected:
    virtual void print(std::ostream &os) const;

    std::filesystem::path input_path;
    std::filesystem::path output_path;
    bool verbose;
  };

  class TrainingControllerParameters : public ControllerParameters {
  public:
    TrainingControllerParameters(const std::filesystem::path &input_path,
                                 const std::filesystem::path &output_path, size_t max_epoch,
                                 size_t batch_size, bool verbose = false);

    [[nodiscard]] size_t getMaxEpoch() const { return max_epoch; }
    void setMaxEpoch(size_t e) { max_epoch = e; }

    [[nodiscard]] size_t getBatchSize() const { return batch_size; }
    void setBatchSize(size_t e) { batch_size = e; }

  private:
    void print(std::ostream &os) const override;

    size_t max_epoch;

    /*Size of the training batch, not to be confused with the size of the batch
                         used during gradient descent*/
    size_t batch_size;
  };

}   // namespace control
