#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "ActivationFunction.hpp"
#include "CNNLayer.hpp"
#include "CNNStorageBP.hpp"
#include "CNNTopologyLayer.hpp"

namespace nnet {

  /**
   * @brief Class to describe the topology of cnn
   */
  class CNNTopology {
    typedef typename std::vector<std::shared_ptr<CNNTopologyLayer>>::const_iterator const_iterator;
    friend std::ostream &operator<<(std::ostream &os, const CNNTopology &topology);
    friend const CNNTopology stringToTopology(const std::string &str);

  public:
    CNNTopology();
    explicit CNNTopology(const std::pair<size_t, size_t> &inputSize);

    CNNTopology(const CNNTopology &) = default;
    CNNTopology(CNNTopology &&) = default;

    CNNTopology &operator=(const CNNTopology &) = default;
    CNNTopology &operator=(CNNTopology &&) = default;

    [[nodiscard]] const_iterator cbegin() const { return layers.begin(); }
    [[nodiscard]] const_iterator begin() const { return layers.begin(); }

    [[nodiscard]] const_iterator cend() const { return layers.end(); }
    [[nodiscard]] const_iterator end() const { return layers.end(); }

    /**
     * @brief Operator to access of one topology layer
     * @param index of layer
     * @return topology layer
     */
    const std::shared_ptr<CNNTopologyLayer> &operator()(size_t index) const;


    [[nodiscard]] const af::ActivationFunctionType getActivationFunction() const {
      return activationFunction;
    }

    [[nodiscard]] const std::pair<size_t, size_t> &getInputSize() const { return inputSize; }
    [[nodiscard]] const size_t getDepth() const { return layers.size(); }

    [[nodiscard]] const size_t getNBranchFinal() const { return n_branch_final; }
    [[nodiscard]] size_t getCNNOutputSize() const {
      return n_branch_final * layers.back()->getOutputSize().first *
             layers.back()->getOutputSize().second;
    }

    /**
     * @brief Convert the topology layer into vector of CNNLayer
     * @return Converted layer topology into vector of CNNLayer
     */
    [[nodiscard]] std::vector<std::unique_ptr<CNNLayer>> convertToLayer() const;

    /**
     * @brief Convert the topology layer into vector of CNNStorageBP
     * @return Converted layer topology into vector of CNNStorageBP
     */
    [[nodiscard]] std::vector<std::unique_ptr<CNNStorageBP>> convertToStorage() const;

  private:
    /**
     * @brief Add a convolution pooling to topology
     * @param inputSize Input image size
     * @param features Number of feature of convolution
     * @param filterSize Filter size
     * @param aFunction Activation function use in convolution
     * @param nbranch Number of branch
     */
    void addConvolution(const std::pair<size_t, size_t> &inputSize, const size_t features,
                        const std::pair<size_t, size_t> &filterSize,
                        const af::ActivationFunctionType aFunction, const size_t nbranch);

    /**
     * Add a pooling layer to topology
     * @param inputSize Input image size
     * @param poolingType Type of pooling
     * @param poolSize Pooling size
     * @param nbranch Number of branch
     */
    void addPooling(const std::pair<size_t, size_t> &inputSize, const PoolingType poolingType,
                    const std::pair<size_t, size_t> &poolSize, const size_t nbranch);

  private:
    std::pair<size_t, size_t> inputSize;
    std::vector<std::shared_ptr<CNNTopologyLayer>> layers;
    af::ActivationFunctionType activationFunction;
    size_t n_branch_final;
  };

  /**
   * @brief Convert a string into topology
   * @param str String to convert
   * @return Topology generate with the string
   */
  const CNNTopology stringToTopology(const std::string &str);

  std::ostream &operator<<(std::ostream &os, const CNNTopologyLayer &nn);

}   // namespace nnet