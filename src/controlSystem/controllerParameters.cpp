
#include "controlSystem/controllerParameters.hpp"

#include <utility>

namespace control {

  std::ostream &operator<<(std::ostream &os, const ControllerParameters &cp) {
    cp.print(os);
    return os;
  }

  ControllerParameters::ControllerParameters(std::filesystem::path input_path,
                                             std::filesystem::path output_path, bool verbose)
      : input_path(std::move(input_path)), output_path(std::move(output_path)), verbose(verbose) {}

  void ControllerParameters::print(std::ostream &os) const {
    os << "Controller Parameters: " << std::endl;
    os << "\tinput_path: " << input_path << std::endl;
    os << "\toutput_path: " << output_path << std::endl;
    os << "\tverbose: " << verbose << std::endl;
  }

  TrainingControllerParameters::TrainingControllerParameters(
          const std::filesystem::path &input_path, const std::filesystem::path &output_path,
          size_t max_epoch, size_t batch_size, bool verbose)
      : ControllerParameters(input_path, output_path, verbose), max_epoch(max_epoch),
        batch_size(batch_size) {}

  void TrainingControllerParameters::print(std::ostream &os) const {
    ControllerParameters::print(os);
    os << "\tmax_epoch: " << max_epoch << std::endl;
    os << "\tbatch_size: " << batch_size << std::endl;
  }

}   // namespace control