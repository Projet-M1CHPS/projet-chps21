#include <controlSystem/inputSet.hpp>

namespace control {

  std::ostream &operator<<(std::ostream &os, InputSet const &set) {
    set.print(os);
    return os;
  }

  void InputSet::print(std::ostream &os) const {
    os << "InputSet: " << inputs.size() << " elements" << std::endl;
    os << std::endl;
  }

}   // namespace control
