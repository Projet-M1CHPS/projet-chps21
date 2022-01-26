#pragma once

#include "Matrix.hpp"
#include "Model.hpp"
#include <vector>

namespace nnet {

  class ModelOptimizer {
  public:
    virtual void optimize(const std::vector<math::FloatMatrix> &inputs,
                          const std::vector<math::FloatMatrix> &targets) = 0;

    virtual ~ModelOptimizer() = default;

    virtual void update() = 0;
  };

}   // namespace nnet
