#pragma once

#include "ActivationFunction.hpp"
#include "CNNTopology.hpp"
#include "Matrix.hpp"

namespace cnnet {

  class CNN {
  public:
    CNN() = default;

    CNN(const CNN &other) = default;
    CNN(CNN &&other) noexcept = default;
    CNN &operator=(const CNN &) = default;
    CNN &operator=(CNN &&other) noexcept = default;

    void setTopology(CNNTopology const &topology);
    [[nodiscard]] CNNTopology const &getTopology() const { return topology; }

    void setActivationFunction(af::ActivationFunctionType type);
    [[nodiscard]] const std::vector<af::ActivationFunctionType> &getActivationFunctions() const;

    void randomizeWeight();

    const math::FloatMatrix &predict(math::FloatMatrix const &input);

  private:
    CNNTopology topology;

    std::vector<std::unique_ptr<CNNLayer>> layers;
    std::vector<FloatMatrix> layerMatrix;

    af::ActivationFunctionType activation_function;
  };

}   // namespace cnnet