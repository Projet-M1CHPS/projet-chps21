#pragma once

#include "Matrix.hpp"
#include "Model.hpp"
#include <vector>

namespace nnet {

  enum class ModelOptimizerType { stochastic, batch, minibatch };

  template<typename real = float>
  class ModelOptimizer {
  public:
    virtual void optimize(const std::vector<math::Matrix<real>> &inputs,
                          const std::vector<math::Matrix<real>> &targets) = 0;

    virtual ~ModelOptimizer() = default;

    virtual void update() = 0;

    virtual void setModel(Model<real> &model) = 0;


  private:
  };

}   // namespace nnet
