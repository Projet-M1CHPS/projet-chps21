#pragma once

#include "Matrix.hpp"
#include "Model.hpp"
#include <vector>

namespace nnet {

  template<typename real = float>
  class ModelOptimizer {
  public:
    virtual void optimize(const std::vector<math::Matrix<real>> &inputs,
                          const std::vector<math::Matrix<real>> &targets) = 0;

    virtual ~ModelOptimizer() = default;

    virtual void update() = 0;
  };

}   // namespace nnet
