#pragma once

#include "Matrix.hpp"
#include "tscl.hpp"
#include <filesystem>
#include <iostream>

namespace nnet {
  class Model {
  public:
    Model() = default;
    ~Model() = default;

    Model(Model const &other) = delete;
    Model &operator=(Model const &other) = delete;

    Model(Model &&other) noexcept = default;
    Model &operator=(Model &&other) noexcept = default;

    virtual math::FloatMatrix predict(math::FloatMatrix const &input) = 0;
  };

}   // namespace nnet