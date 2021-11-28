
#include "controlSystem/RunConfiguration.hpp"
#include "controlSystem/ImageCache.hpp"
#include <search.h>
#include <utility>

namespace control {

  RunConfiguration::RunConfiguration()
      :
        precision(nnet::FloatingPrecision::float32), working_dir("series"), mode(trainingMode) {}
  RunConfiguration::RunConfiguration(std::filesystem::path input_path,
                                     std::filesystem::path working_dir,
                                     const std::filesystem::path &target_dir)
      : RunConfiguration() {
    this->input_path = std::move(input_path);
    this->working_dir = std::move(working_dir);

    // If the target dir points to an existing dir or an absolute location, just assign it
    if (target_dir.is_absolute())
      this->target_dir = target_dir;
    else {   // Else, target dir shall be a sub-folder in the working_dir
      this->target_dir = this->working_dir;
      this->target_dir.append(target_dir.c_str());
    }
  }
  bool RunConfiguration::operator==(RunConfiguration const &other) const { return false; }

  std::filesystem::path const &RunConfiguration::getWorkingDirectory() const { return working_dir; }

  std::filesystem::path const &RunConfiguration::getTargetDirectory() const { return target_dir; }

  std::filesystem::path const &RunConfiguration::getInputPath() const { return input_path; }

  std::vector<image::transform::TransformType> const &RunConfiguration::getTransformations() const {
    return transformations;
  }
  std::vector<af::ActivationFunctionType> const &RunConfiguration::getActivationFunctions() const {
    return activation_functions;
  }
  nnet::FloatingPrecision RunConfiguration::getFPPrecision() const { return precision; }

  std::vector<size_t> const &RunConfiguration::getTopology() const { return topology; }
}   // namespace control