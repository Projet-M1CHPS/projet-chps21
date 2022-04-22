#pragma once
#include "BatchOptimizationScheduler.hpp"
#include "BatchProgression.hpp"
#include "ParallelScheduler.hpp"
#include <mpi.h>

namespace nnet {

  /**
   * @brief An MPI implementation of the ParallelScheduler
   * Model is gather, averaged, and broadcast to all nodes for each epoch
   */
  class MPIParallelScheduler : public ParallelScheduler {
  public:
    class Builder;
    using ParallelScheduler::ParallelScheduler;

    MPIParallelScheduler(const ParallelScheduler &other) = delete;
    explicit MPIParallelScheduler(ParallelScheduler &&other)
        : ParallelScheduler(std::move(other)) {}

    /**
     * @brief Runs the optimizer for a single epoch with MPI synchronization. Batch displacement is
     * determined by the policy used during the construction of the ParallelScheduler.
     */
    void run() override;

    /** Getter for the work sizes of every process
     */
    [[nodiscard]] std::vector<size_t> getWorkSizes() const { return processes_work_sizes; }

    /** Getter for the work sizes of a specific process
     */
    [[nodiscard]] size_t getWorkSize(int rank) const { return processes_work_sizes.at(rank); }

  private:
    std::vector<size_t> processes_work_sizes;

    std::vector<MPI_Comm> synchronizeGlobalWorkSize();
  };

  class MPIParallelScheduler::Builder : public ParallelScheduler::Builder {
  public:
    using ParallelScheduler::Builder::Builder;

    [[nodiscard]] std::unique_ptr<ParallelScheduler> build() const override;
  };
}   // namespace nnet