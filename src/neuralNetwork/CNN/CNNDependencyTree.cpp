#include "CNNDependencyTree.hpp"

namespace nnet {
  void CNNDependencyTree::build(const CNNTopology &topology) {
    if (root) { assert(false && "Already built"); }
    if (not topology.getDepth()) { throw std::runtime_error("Topology is empty"); }

    root = std::make_shared<CNNNode>(0, 0, nullptr, topology(0)->convertToLayer());
    if (topology.getDepth() == 1) return;

    std::stack<CNNNode*> stack, next_stack;
    stack.push(root.get());

    size_t global_id = 1;


    for (size_t i = 1; i < topology.getDepth() - 1; i++) {
      while (not stack.empty()) {
        CNNNode* node = stack.top();
        stack.pop();

        for (size_t j = 0; j < topology(i - 1)->getFeatures(); j++) {
          CNNNode new_child(j, global_id++, node, topology(i)->convertToLayer());
          node->addChild(std::move(new_child));
        }
        for (auto &j : node->getChildren()) { next_stack.push(&j); }
      }
      stack.swap(next_stack);
      next_stack = std::stack<CNNNode*>();
    }

    const size_t last = topology.getDepth() - 1;

    while (not stack.empty()) {
      CNNNode* node = stack.top();
      stack.pop();

      for (size_t j = 0; j < topology(last - 1)->getFeatures(); j++) {
        CNNNode new_child(j, global_id++, node, topology(last)->convertToLayer());
        leaves.push_back(std::make_shared<CNNNode>(new_child));
        node->addChild(std::move(new_child));
      }
    }
  }

  std::ostream &operator<<(std::ostream &os, const CNNDependencyTree &tree) {
    std::stack<std::shared_ptr<CNNNode>> stack;
    stack.push(tree.getRoot());

    while (not stack.empty()) {
      std::shared_ptr<CNNNode> node = stack.top();
      stack.pop();

      std::cout << "global/local id [" << node->getGlobalId() << "][" << node->getLocalId() << "]";
      if (node->getChildren().empty()) {
        os << " no leave\n";
      } else {
        os << " leave(s)";
        for (auto &i : node->getChildren()) { os << " [" << i.getGlobalId() << "]"; }
        os << "\n";
      }
      for (auto &child : node->getChildren()) { stack.push(std::make_shared<CNNNode>(child)); }
    }

    return os;
  }

}   // namespace nnet