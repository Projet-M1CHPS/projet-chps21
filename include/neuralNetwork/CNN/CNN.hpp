#pragma once

#include "ActivationFunction.hpp"
#include "CNNTopology.hpp"
#include "Matrix.hpp"
#include "math/clFMatrix.hpp"
#include "openclUtils/clWrapper.hpp"
#include <utility>

namespace nnet {

  /**
   * @brief Convolutional Neural Network without the mlp
   */
  class CNN {
  public:
    CNN() = default;

    CNN(const CNN &other) = delete;
    CNN(CNN &&other) = delete;

    CNN &operator=(const CNN &) = delete;
    CNN &operator=(CNN &&other) = delete;

    /**
     * @brief Copying CNNLayers
     * @return std::vector<std::unique_ptr<CNNLayer>>
     */
    std::vector<std::unique_ptr<CNNLayer>> copyLayers();

    /**
     * @brief Set the Topology object
     *
     * @param topology
     */
    void setTopology(CNNTopology const &topology);

    /**
     * @brief Get the Topology object
     *
     * @return CNNTopology const&
     */
    [[nodiscard]] CNNTopology const &getTopology() const { return topology; }

    [[nodiscard]] const std::vector<std::unique_ptr<CNNLayer>> &getLayers() const { return layers; }

    /**
     * @brief Randomize the weights of the network
     */
    void randomizeWeight();

    /**
     * @brief
     *
     * @param queue Queue uses for OpenCL
     * @param input Input tensor
     * @return math::clFTensor Output tensor
     */
    math::clFTensor predict(cl::CommandQueue &queue, math::clFTensor const &input);

    void printWeights()
    {
      for(auto& layer : layers)
      {
        if(layer->hasWeight())
        {
          std::cout << "filter : " << layer->getWeight() << std::endl;
        }
      }
    }

  private:
    CNNTopology topology;
    std::vector<std::unique_ptr<CNNLayer>> layers;
  };

  /**
   * @brief Reorganize the output tensor of CNN for MLP
   *
   * @param queue Queue uses for openCL
   * @param tensor Tensor to reorganize
   * @param nInput Number of input
   * @param nBranch Number of branch
   */
  void reorganizeForward(cl::CommandQueue &queue, math::clFTensor &tensor, const size_t nInput,
                         const size_t nBranch);

  /**
   * @brief Reorganize the output tensor of MLP for CNN
   *
   * @param queue Queue uses for openCL
   * @param tensor Tensor to reorganize
   * @param nInput Number of input
   * @param nBranch Number of branch
   * @param size Size of one output of CNN
   */
  void reorganizeBackward(cl::CommandQueue &queue, math::clFTensor &tensor, const size_t nInput,
                          const size_t nBranch, const std::pair<size_t, size_t> size);

}   // namespace nnet