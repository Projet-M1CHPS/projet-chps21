#pragma once

#include "CNNLayer.hpp"
#include <memory>
#include <vector>

namespace nnet {

  class CNNNode {
  public:
    CNNNode(const size_t node_local_id, const size_t node_global_id,
            const std::shared_ptr<CNNNode> &node_father,
            const std::shared_ptr<CNNLayer> &node_layer)
        : local_id(node_local_id), global_id(node_global_id), father(node_father), layer(node_layer) {}

    [[nodiscard]] const size_t getLocalId() const { return local_id; }
    [[nodiscard]] const size_t getGlobalId() const { return global_id; }

    [[nodiscard]] const std::shared_ptr<CNNNode> &getFather() const { return father; }
    void setFather(std::shared_ptr<CNNNode> &node_father) { father = node_father; }

    [[nodiscard]] const std::vector<CNNNode> &getChildren() const { return children; }

    [[nodiscard]] CNNLayer *getLayer() const { return layer.get(); }

    void addChild(CNNNode &&child);


  private:
    const size_t local_id;
    const size_t global_id;
    std::shared_ptr<CNNNode> father;
    std::vector<CNNNode> children;

    std::shared_ptr<CNNLayer> layer;
  };

}   // namespace nnet