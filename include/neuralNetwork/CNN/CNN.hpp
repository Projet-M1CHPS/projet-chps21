#pragma once

#include "ActivationFunction.hpp"
#include "CNNTopology.hpp"
#include "Matrix.hpp"
#include "clUtils/clFMatrix.hpp"
#include "clUtils/clWrapper.hpp"
#include "CNNDependencyTree.hpp"
#include <stack>
#include <utility>

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
      // TODO: check if this is correct
      return 0;
    }

    void randomizeWeight();

    void predict(math::clFMatrix const &input, math::clFMatrix &output);

  private:
    CNNTopology topology;

    CNNDependencyTree tree;
  };

}   // namespace cnnet