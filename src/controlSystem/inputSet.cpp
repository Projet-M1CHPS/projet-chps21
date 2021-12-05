#include <controlSystem/inputSet.hpp>
#include <utility>

namespace control {

  void InputSet::append(std::filesystem::path path, math::Matrix<float> &&mat) {
    inputs_path.push_back(std::move(path));
    inputs.push_back(std::move(mat));
  }

  std::ostream &operator<<(std::ostream &os, InputSet const &set) {
    for (auto const &input : set.inputs_path) { os << "\t" << input << std::endl; }
    return os;
  }


}   // namespace control
