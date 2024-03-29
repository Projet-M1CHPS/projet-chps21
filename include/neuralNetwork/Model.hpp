#pragma once

#include "math/Matrix.hpp"
#include "math/clFMatrix.hpp"
#include "math/clFTensor.hpp"
#include "openclUtils//clWrapper.hpp"
#include <filesystem>
#include <utility>

namespace nnet {
  /**
   * @brief Base class for all neural network models.
   */
  class Model {
  public:
    Model() = default;
    virtual ~Model() = default;

    //
    // Models should not be trivially copyable
    //
    Model(Model const &other) = delete;
    Model &operator=(Model const &other) = delete;

    Model(Model &&other) noexcept = default;
    Model &operator=(Model &&other) noexcept = default;

    /**
     * @brief Runs the model
     * @param input The model's input
     * @return The model's output
     */
    [[nodiscard]] virtual math::clFMatrix predict(cl::CommandQueue &queue, math::clFMatrix const &input) const = 0;

    [[nodiscard]] virtual math::clFTensor predict(cl::CommandQueue &queue, math::clFTensor const &input) const = 0;

    /**
     * @brief Save the model to the given path
     * @param path The path where the model should be saved
     * Note that the model saving format may use multiple files
     * @return True if the model was saved successfully, false otherwise
     */
    virtual bool save(const std::filesystem::path &path) const = 0;

    /**
     * @brief Replace this model by one loaded from the given path
     * If loading fails, this model will be left unchanged
     * @param path The path where the model should be loaded
     * @return True if the model was loaded successfully, false otherwise
     */
    virtual bool load(const std::filesystem::path &path) = 0;
  };

}   // namespace nnet