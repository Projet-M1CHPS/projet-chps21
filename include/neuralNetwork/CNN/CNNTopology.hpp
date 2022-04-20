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

    [[nodiscard]] std::vector<std::unique_ptr<CNNLayer>> convertToLayer() const;
    [[nodiscard]] std::vector<std::unique_ptr<CNNStorageBP>> convertToStorage() const;

  private:
    void addConvolution(const std::pair<size_t, size_t> &inputSize, const size_t features,
                        const std::pair<size_t, size_t> &filterSize,
                        const af::ActivationFunctionType aFunction, const size_t nbranch);

    void addPooling(const std::pair<size_t, size_t> &inputSize, const PoolingType poolingType,
                    const std::pair<size_t, size_t> &poolSize, const size_t nbranch);

  private:
    std::pair<size_t, size_t> inputSize;
    std::vector<std::shared_ptr<CNNTopologyLayer>> layers;
    af::ActivationFunctionType activationFunction;
    size_t n_branch_final;
  };

  const CNNTopology stringToTopology(const std::string &str);

  std::ostream &operator<<(std::ostream &os, const CNNTopologyLayer &nn);

}   // namespace nnet