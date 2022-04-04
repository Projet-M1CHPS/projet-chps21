#pragma once

#include "CNNNode.hpp"
#include "CNNTopology.hpp"
#include <stack>
#include <vector>

namespace nnet {

  class CNNDependencyTree {
    friend std::ostream &operator<<(std::ostream &os, const CNNTopology &topology);

  public:
    CNNDependencyTree() = default;
    [[maybe_unused]] explicit CNNDependencyTree(const CNNTopology &topology) { build(topology); }

    void build(const CNNTopology &topology);

    [[nodiscard]] const std::shared_ptr<CNNNode> &getRoot() const { return root; }
    [[nodiscard]] const std::vector<std::shared_ptr<CNNNode>> &getLeaves() const { return leaves; }

  private:
    std::shared_ptr<CNNNode> root;
    std::vector<std::shared_ptr<CNNNode>> leaves;
  };

  std::ostream &operator<<(std::ostream &os, const CNNDependencyTree &tree);

}   // namespace nnet