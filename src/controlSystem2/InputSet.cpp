#include "InputSet.hpp"

namespace control {

  void InputSet::append(std::vector<Sample> &&sample, math::clFTensor &&tensor) {
    if (sample.size() != tensor.getZ()) {
      throw std::runtime_error("InputSet::append: sample and tensor size mismatch");
    }

    for (auto &s : sample) {
      samples.push_back(s);
    }
    tensors.push_back(std::move(tensor));
  }

}   // namespace control