#pragma once
#include "InputSet.hpp"


namespace control {
  class TrainingCollection {
  public :

  private :
    InputSet training_set;
    InputSet eval_set;
  };

}   // namespace control
