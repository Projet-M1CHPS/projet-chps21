
#include "CTCollection.hpp"
#include "tscl.hpp"
#include <clUtils/clWrapper.hpp>
#include <functional>


namespace control::classifier {


  CTCollection::CTCollection(std::shared_ptr<CClassLabelSet> classes)
      : class_list(std::move(classes)) {}

  std::ostream &operator<<(std::ostream &os, CTCollection const &set) {
    os << "Classifier training set: " << std::endl;
    os << "\tTraining set contains " << set.training_set.size() << " elements" << std::endl;
    os << "\tEvaluation set contains " << set.eval_set.size() << " elements" << std::endl;

    os << "Classes: " << std::endl;
    os << *set.class_list << std::endl;

    return os;
  }
}   // namespace control::classifier