#pragma once
#include "BackpropStorage.hpp"
#include "MLPerceptron.hpp"
#include "Matrix.hpp"

namespace nnet {

  /**
   * @brief Base class for all optimization
   */
  class Optimization {
  public:
    Optimization() = default;
    virtual ~Optimization() = default;

    /**
     *
     * @param storage
     */
    virtual void optimize(BackpropStorage &storage, utils::clWrapper &wrapper,
                          cl::CommandQueue &queue) = 0;

    /**
     * @brief Some optimization require to be updated after each epoch
     * We provide an empty declaration here to avoid to have to implement it where not needed
     */
    virtual void update(){};
  };
}   // namespace nnet