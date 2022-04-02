#include "CNNNode.hpp"

namespace nnet {
  void CNNNode::addChild(CNNNode&& child)
  {
    children.push_back(child);
  }
}