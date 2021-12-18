#include "classifierClass.hpp"

void CClassLabelSet::append(ClassLabel label) { labels.emplace(label.getId(), std::move(label)); }

std::ostream &operator<<(std::ostream &os, const ClassLabel &label) {
  os << "Class: " << label.getId() << ": " << label.getName();
  return os;
}


std::ostream &operator<<(std::ostream &os, const CClassLabelSet &label) {
  for (auto &l : label) { os << l.second << std::endl; }
  return os;
}