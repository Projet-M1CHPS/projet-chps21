#pragma once

#include "ActivationFunction.hpp"
#include "CNNTopology.hpp"
#include "Matrix.hpp"
#include "clUtils/clFMatrix.hpp"
#include "clUtils/clWrapper.hpp"

namespace nnet {

  class CNN {
  public:
    CNN() = default;

    CNN(const CNN &other) = default;
    CNN(CNN &&other) noexcept;
    CNN &operator=(const CNN &) = default;
    CNN &operator=(CNN &&other) noexcept;

    void setTopology(CNNTopology const &topology);
    [[nodiscard]] CNNTopology const &getTopology() const { return topology; }

    [[nodiscard]] size_t getOutputSize() const {
      return layerMatrix.back().size() * layerMatrix.back()[0].getCols() *
             layerMatrix.back()[0].getRows();
    }

    [[nodiscard]] const std::vector<std::vector<std::shared_ptr<CNNLayer>>> &getLayers() const { return layers; }

    void randomizeWeight();

    void predict(math::clFMatrix const &input, math::clFMatrix &output);

  private:
    CNNTopology topology;

    std::vector<std::vector<std::shared_ptr<CNNLayer>>> layers;
    std::vector<std::vector<FloatMatrix>> layerMatrix;
  };

}   // namespace cnnet