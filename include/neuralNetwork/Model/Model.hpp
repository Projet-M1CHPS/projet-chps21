#pragma once

#include "Matrix.hpp"
#include "tscl.hpp"
#include <filesystem>
#include <iostream>

namespace nnet {

  /** @brief Base interface for neural network Models
   *
   * @tparam real
   */
  template<typename real = float, typename = std::enable_if<std::is_floating_point_v<real>>>
  class Model {
  public:
    Model() = default;
    ~Model() = default;

    Model(Model const &other) = delete;
    Model &operator=(Model const &other) = delete;

    Model(Model &&other) noexcept = default;
    Model &operator=(Model &&other) noexcept = default;

    virtual math::Matrix<real> predict(math::Matrix<real> const &input) = 0;
  };

}   // namespace nnet