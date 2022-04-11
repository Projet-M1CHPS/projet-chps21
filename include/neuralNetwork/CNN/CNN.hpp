#pragma once

#include "ActivationFunction.hpp"
#include "CNNTopology.hpp"
#include "Matrix.hpp"
#include "math/clFMatrix.hpp"
#include "openclUtils/clWrapper.hpp"
#include <utility>

namespace nnet {

  using namespace math;

  class CNN {
  public:
    CNN() = default;

    CNN(const CNN &other) = default;
    CNN(CNN &&other) = default;
    CNN &operator=(const CNN &) = default;
    CNN &operator=(CNN &&other) = default;

    void setTopology(CNNTopology const &topology);
    [[nodiscard]] CNNTopology const &getTopology() const { return topology; }

    [[nodiscard]] const std::vector<std::shared_ptr<CNNLayer>> &getLayers() const { return layers; }

    [[nodiscard]] size_t getOutputSize() const {
      // TODO : warning
      assert(0 && "A voir si on en a besoin");
      return 0;
    }

    void randomizeWeight();

    clFTensor predict(clFTensor const &input);

  public:
    CNNTopology topology;
    std::vector<std::shared_ptr<CNNLayer>> layers;
  };

}   // namespace nnet