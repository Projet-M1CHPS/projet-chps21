#pragma once

#include "ActivationFunction.hpp"
#include "CNNTopology.hpp"
#include "Matrix.hpp"

namespace cnnet {

  class CNN {
  public:
    CNN() = default;

    CNN(const CNN &other) = default;
    CNN(CNN &&other) noexcept;
    CNN &operator=(const CNN &) = default;
    CNN &operator=(CNN &&other) noexcept;

    void setTopology(CNNTopology const &topology);
    [[nodiscard]] CNNTopology const &getTopology() const { return topology; }

    void setActivationFunction(af::ActivationFunctionType type) { activation_function = type; }
    const af::ActivationFunctionType getActivationFunctions() const { return activation_function; }

    const size_t getOutputSize() const {
      return layerMatrix.back().size() * layerMatrix.back()[0].getCols() *
             layerMatrix.back()[0].getRows();
    }

    const std::vector<std::vector<std::shared_ptr<CNNLayer>>> &getLayers() const { return layers; }

    void randomizeWeight();

    void predict(math::FloatMatrix const &input, math::FloatMatrix &output);

  private:
    CNNTopology topology;

    std::vector<std::vector<std::shared_ptr<CNNLayer>>> layers;
    std::vector<std::vector<FloatMatrix>> layerMatrix;

    af::ActivationFunctionType activation_function;
  };

}   // namespace cnnet