#pragma once

#include "ActivationFunction.hpp"
#include "CNNStorageBP.hpp"
#include "Matrix.hpp"
#include "math/clFMatrix.hpp"
#include "math/clFTensor.hpp"
#include "openclUtils/clWrapper.hpp"
#include <iostream>


namespace nnet {
  enum class LayerType { CONVOLUTION, POOLING };
  enum class PoolingType { MAX, AVERAGE };

  /**
   * @brief Base class of cnn layer
   */
  class CNNLayer {
  public:
    explicit CNNLayer(const std::pair<size_t, size_t> outputSize);
    virtual ~CNNLayer() = default;

    /**
     * @brief copying a layer
     * @return the copied layer
     */
    virtual std::unique_ptr<CNNLayer> copy() const = 0;

    /**
     * @brief Check if layer have weight
     * @return true if layer have weight, otherwise false
     */
    [[nodiscard]] virtual bool hasWeight() const { return false; }

    /**
     * @brief Getter to retrieve weight
     * @return return weight if they exist otherwise throw an assertion
     */
    virtual math::clFTensor &getWeight() {
      throw std::runtime_error("CNNLayer: Tried to acces weight in a layer without one ");
    }

    /**
     * @brief Getter to retrieve weight
     * @return return weight if they exist otherwise throw an assertion
     */
    virtual const math::clFTensor &getWeight() const {
      throw std::runtime_error("CNNLayer: Tried to acces weight in a layer without one ");
    }

    virtual void setWeight(const math::clFTensor &weights);

    /**
     * @brief Compute layer operation for prediction of cnn
     * @param queue Queue used to make the computations
     * @param input Input tensor
     * @return Tensor transform by layer operation
     */
    virtual math::clFTensor compute(cl::CommandQueue &queue, const math::clFTensor &input) = 0;

    /**
     * @brief Compute layer operation for forward of cnn
     * @param queue Queue used to make the computations
     * @param input Input tensor
     * @param storage Backpack with data to learn
     * @return Tensor transform by layer operation
     */
    virtual math::clFTensor computeForward(cl::CommandQueue &queue, const math::clFTensor &input,
                                           CNNStorageBP &storage) = 0;

    /**
     * @brief Compute layer operation for backward of cnn
     * @param queue Queue used to make the computations
     * @param input Input tensor
     * @param storage Backpack with data to learn
     * @return Tensor transform by layer operation
     */
    virtual math::clFTensor computeBackward(cl::CommandQueue &queue, const math::clFTensor &input,
                                            CNNStorageBP &storage) = 0;

    [[nodiscard]] const std::pair<size_t, size_t> getOutputSize() const { return outputSize; }

  protected:
    const std::pair<size_t, size_t> outputSize;
  };

  /**
   * @brief Implementation of convolutional layer
   */
  class CNNConvolutionLayer final : public CNNLayer {
  public:
    // nFilter nombre de kernel par branche
    CNNConvolutionLayer(const std::pair<size_t, size_t> outputSize,
                        const std::pair<size_t, size_t> sizeFilter, const size_t nFilter,
                        const af::ActivationFunctionType aFunction, const size_t nBranch);

    CNNConvolutionLayer(const CNNConvolutionLayer &other);
    ~CNNConvolutionLayer() override = default;

    /**
     * @brief copying a layer
     * @return the copied layer
     */
    [[nodiscard]] std::unique_ptr<CNNLayer> copy() const override;

    /**
     * @brief Check if layer have weight
     * @return true if layer have weight, otherwise false
     */
    [[nodiscard]] bool hasWeight() const override { return true; }

    /**
     * @brief Getter to retrieve weight
     * @return return weight if they exist otherwise throw an assertion
     */
    math::clFTensor &getWeight() override { return filters; }
    // TODO : hum interressant plus de function pour faire la meme chose ?

    /**
     * @brief Getter to retrieve weight
     * @return return weight if they exist otherwise throw an assertion
     */
    const math::clFTensor &getWeight() const override { return filters; }
    void setWeight(const math::clFTensor &weights) override;

    [[nodiscard]] const math::clFTensor &getFilter() const { return filters; }
    [[nodiscard]] math::clFTensor &getFilter() { return filters; }

    /**
     * @brief Compute layer operation for prediction of cnn
     * @param queue Queue used to make the computation
     * @param input Input tensor
     * @return Tensor transform by layer operation
     */
    math::clFTensor compute(cl::CommandQueue &queue, const math::clFTensor &input) override;

    /**
     * @brief Compute layer operation for forward of cnn
     * @param queue Queue used to make the computations
     * @param input Input tensor
     * @param storage Backpack with data to learn
     * @return Tensor transform by layer operation
     */
    math::clFTensor computeForward(cl::CommandQueue &queue, const math::clFTensor &input,
                                   CNNStorageBP &storage) override;

    /**
     * @brief Compute layer operation for backward of cnn
     * @param queue Queue used to make the computations
     * @param input Input tensor
     * @param storage Backpack with data to learn
     * @return Tensor transform by layer operation
     */
    math::clFTensor computeBackward(cl::CommandQueue &queue, const math::clFTensor &input,
                                    CNNStorageBP &storage) override;

  private:
    /**
     * @brief Compute the error on filter
     * @param queue Queue used to make the computations
     * @param input Input tensor
     * @param storage Backpack with data to learn
     * @return Error Tensor of filter
     */
    math::clFTensor computeErrorFilter(cl::CommandQueue &queue, const math::clFTensor &input,
                                       CNNStorageBPConvolution &storage);

    /**
     * @brief Compute the error on input
     * @param queue Queue used to make the computations
     * @param input Input tensor
     * @param storage Backpack with data to learn
     * @return Error Tensor of input
     */
    math::clFTensor computeErrorInput(cl::CommandQueue &queue, const math::clFTensor &input,
                                      CNNStorageBPConvolution &storage);

  private:
    const size_t n_branch;
    const size_t n_filter;

    math::clFTensor filters;
    af::ActivationFunctionType a_function;
  };


  /**
   * @brief Base class of pooliong layer
   */
  class CNNPoolingLayer : public CNNLayer {
  public:
    CNNPoolingLayer(const std::pair<size_t, size_t> outputSize,
                    const std::pair<size_t, size_t> PoolSize);
    ~CNNPoolingLayer() override = default;

  protected:
    const std::pair<size_t, size_t> poolingSize;
  };

  /**
   * @brief Implementation of max pooliong layer
   */
  class CNNMaxPoolingLayer final : public CNNPoolingLayer {
  public:
    CNNMaxPoolingLayer(const std::pair<size_t, size_t> outputSize,
                       const std::pair<size_t, size_t> PoolSize);
    CNNMaxPoolingLayer(const CNNMaxPoolingLayer &other);
    ~CNNMaxPoolingLayer() override = default;

    /**
     * @brief copying a layer
     * @return the copied layer
     */
    std::unique_ptr<CNNLayer> copy() const override;

    /**
     * @brief Compute layer operation for prediction of cnn
     * @param queue Queue used to make the computations
     * @param input Input tensor
     * @return Tensor transform
     */
    math::clFTensor compute(cl::CommandQueue &queue, const math::clFTensor &input) override;

    /**
     * @brief Compute layer operation for forward of cnn
     * @param queue Queue used to make the computations
     * @param input Input tensor
     * @param storage Backpack with data to learn
     * @return Tensor transform by layer operation
     */
    math::clFTensor computeForward(cl::CommandQueue &queue, const math::clFTensor &input,
                                   CNNStorageBP &storage) override;

    /**
     * @brief Compute layer operation for backward of cnn
     * @param queue Queue used to make the computations
     * @param input Input tensor
     * @param storage Backpack with data to learn
     * @return Tensor transform by layer operation
     */
    math::clFTensor computeBackward(cl::CommandQueue &queue, const math::clFTensor &input,
                                    CNNStorageBP &storage) override;
  };

  /**
   * @brief Implementation of average pooliong layer
   */
  class CNNAvgPoolingLayer final : public CNNPoolingLayer {
  public:
    CNNAvgPoolingLayer(const std::pair<size_t, size_t> outputSize,
                       const std::pair<size_t, size_t> PoolSize);
    CNNAvgPoolingLayer(const CNNAvgPoolingLayer &other);
    ~CNNAvgPoolingLayer() override = default;

    /**
     * @brief copying a layer
     * @return the copied layer
     */
    std::unique_ptr<CNNLayer> copy() const override;

    /**
     * @brief Compute layer operation for prediction of cnn
     * @param queue Queue used to make the computations
     * @param input Input tensor
     * @return Tensor transform
     */
    math::clFTensor compute(cl::CommandQueue &queue, const math::clFTensor &input) override;

    /**
     * @brief Compute layer operation for forward of cnn
     * @param queue Queue used to make the computations
     * @param input Input tensor
     * @param storage Backpack with data to learn
     * @return Tensor transform by layer operation
     */
    math::clFTensor computeForward(cl::CommandQueue &queue, const math::clFTensor &input,
                                   CNNStorageBP &storage) override;

    /**
     * @brief Compute layer operation for backward of cnn
     * @param queue Queue used to make the computations
     * @param input Input tensor
     * @param storage Backpack with data to learn
     * @return Tensor transform by layer operation
     */
    math::clFTensor computeBackward(cl::CommandQueue &queue, const math::clFTensor &input,
                                    CNNStorageBP &storage) override;

  private:
    math::clFTensor filter;
  };

}   // namespace nnet