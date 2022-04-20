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

    CNN(const CNN &other) = delete;
    CNN(CNN &&other) = delete;

    CNN &operator=(const CNN &) = delete;
    CNN &operator=(CNN &&other) = delete;

    std::vector<std::unique_ptr<CNNLayer>> copyLayers();

    void setTopology(CNNTopology const &topology);
    [[nodiscard]] CNNTopology const &getTopology() const { return topology; }

    [[nodiscard]] const std::vector<std::unique_ptr<CNNLayer>> &getLayers() const { return layers; }

    [[nodiscard]] size_t getOutputSize() const {
      // TODO : warning
      assert(0 && "A voir si on en a besoin");
      return 0;
    }

    void randomizeWeight();

    clFTensor predict(clFTensor const &input);

  public:
    CNNTopology topology;
    std::vector<std::unique_ptr<CNNLayer>> layers;
  };

  void reorganizeForward(cl::CommandQueue &queue, clFTensor &tensor, const size_t nInput,
                         const size_t nBranch);
  void reorganizeBackward(cl::CommandQueue &queue, clFTensor &tensor, const size_t nInput,
                          const size_t nBranch, const std::pair<size_t, size_t> size);

}   // namespace nnet