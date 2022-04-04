#include "ClassLabel.hpp"

namespace control::classifier {

  void CClassLabelSet::append(ClassLabel label) {
    if (labels.contains(label.getId()))
      throw std::runtime_error("CClassLabelSet: a label already exists with this id");
    labels.emplace(label.getId(), std::move(label));
  }

  std::ostream &operator<<(std::ostream &os, const ClassLabel &label) {
    os << "Class " << label.getId() << ": " << label.getName();
    return os;
  }


  std::ostream &operator<<(std::ostream &os, const CClassLabelSet &label) {
    for (auto &l : label) { os << l.second << std::endl; }
    return os;
  }
}   // namespace control::classifier