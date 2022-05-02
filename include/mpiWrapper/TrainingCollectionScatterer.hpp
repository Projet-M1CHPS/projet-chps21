#pragma once
#include "TrainingCollection.hpp"
#include "clFMatrix.hpp"
#include "clFTensor.hpp"
#include <iostream>
#include <mpi.h>

namespace mpiw {

  /**
   * @brief Helper class to scatter a training collection over multiple processes.
   */
  class TrainingCollectionScatterer {
  public:
    /**
     * @brief Builds a scatter on the given communicator.
     * @param comm The communicator to use for the scatter.
     */
    TrainingCollectionScatterer(MPI_Comm comm);

    /**
     * @brief Scatter a training collection among all the processes in the communicator.
     * Note that currently, each process is expected to have atleast one tensor.
     * @param global_collection The collection to scatter among all the processes.
     * @return The local collection that should be used by the current process.
     */
    control::TrainingCollection scatter(const control::TrainingCollection &global_collection) const;

    /**
     * @brief Receive a training collection from another process.
     * @param source The source process.
     * @return A sub collection that the current process should use
     */
    control::TrainingCollection receive(int source) const;

  private:
    int rank, size;
    MPI_Comm comm;
  };

}   // namespace mpiw
