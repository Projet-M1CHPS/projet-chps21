#pragma once

#include "Matrix.hpp"
#include "Model.hpp"
#include <vector>

namespace nnet {

  /**
   * @brief Base class for all optimizers
   */
  class Optimizer {
  public:
    Optimizer(utils::clWrapper *wrapper) : wrapper(wrapper) {}
    virtual ~Optimizer() = default;

    /**
     * @brief Optimize the model on the given input and target sets
     * @param inputs The inputs to be fed to the model
     * @param targets The corresponding ouputs targets to be fed to the model
     */
    virtual void optimize(const std::vector<math::clFMatrix> &inputs,
                          const std::vector<math::clFMatrix> &targets) = 0;

    /**
     * @brief The optimizer may hold some internal state that needs updating after each epoch
     * This isn't done automatically since the user may repeat the optimization multiple times
     * Before updating
     */
    virtual void update() = 0;

  protected:
    utils::clWrapper *wrapper;
  };

}   // namespace nnet
