#pragma once
#include "TrainingCollection.hpp"
#include "clFMatrix.hpp"
#include "clFTensor.hpp"
#include <iostream>
#include <mpi.h>

namespace mpiw {


  void sendTensor(size_t rank, const math::clFTensor &tensor);
  math::clFTensor receiveTensor(size_t rank);

  void sendVector(size_t rank, const std::vector<> vector);
  std::vector<> receiveVector(size_t rank, const std::vector<> vector);

  void sendString(size_t rank, const std::string &string);
  std::string receiveString(size_t rank);

  class TrainingCollectionScatterer {
  public:
    TrainingCollectionScatterer();

    control::TrainingCollection scatter(control::TrainingCollection global_collection);
    control::TrainingCollection receive();

  private:
    size_t rank, size;
  };

}   // namespace mpiw
