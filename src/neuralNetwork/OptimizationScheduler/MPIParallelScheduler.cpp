#include "MPIParallelScheduler.hpp"
#include "MPIMLPOptimizer.hpp"
#include "math/clFTensor.hpp"
#include <boost/asio/thread_pool.hpp>
#include <future>

using namespace math;
using namespace boost;

namespace nnet {
  namespace {
    /** Creates a new MPI communicator with the given processes (ranks from the World communicator).
     * @param ranks_to_keep
     * @return the new MPI_Comm
     * @warning returned MPI_Comm can be MPI_COMM_NULL
     */
    MPI_Comm newCommWith(const std::vector<int> &ranks_to_keep) {
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      MPI_Comm new_comm;
      MPI_Group world_group;
      MPI_Comm_group(MPI_COMM_WORLD, &world_group);
      MPI_Group new_group;
      MPI_Group_incl(world_group, (int) ranks_to_keep.size(), ranks_to_keep.data(), &new_group);

      MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);
      if (std::any_of(ranks_to_keep.begin(), ranks_to_keep.end(),
                      [rank](int r) { return r == rank; }))
        assert(new_comm != MPI_COMM_NULL);
      else
        assert(new_comm == MPI_COMM_NULL);
      return new_comm;
    }

    /**
     * Creates sub-communicators for each process, according to their work_size.
     * @param work_sizes work sizes for each rank [work_size of 0-th process, ...]
     * @return [{first minimal work_size, communicator to use}, ...]
     */
    std::vector<std::pair<size_t, MPI_Comm>>
    createSubCommunicators(const std::vector<size_t> &work_sizes) {
      std::vector<std::pair<size_t, MPI_Comm>> result;
      std::vector<size_t> work_sizes_copy = work_sizes;
      while (!work_sizes_copy.empty()) {
        std::vector<int> ranks_to_keep;
        size_t current_work_size =
                *std::min_element(work_sizes_copy.begin(), work_sizes_copy.end());

        // Keep only ranks with a greater or equal current_work_size
        for (int i = 0; i < work_sizes.size(); i++)
          if (work_sizes[i] >= current_work_size) ranks_to_keep.push_back(i);
        std::erase_if(work_sizes_copy, [&](size_t s) { return s <= current_work_size; });

        // Create new communicator
        MPI_Comm new_comm = newCommWith(ranks_to_keep);
        result.emplace_back(current_work_size, new_comm);
      }

      return result;
    }

    std::vector<std::pair<size_t, MPI_Comm>> synchronizeGlobalWorkSize(size_t local_work_size) {
      int n_process = 0;
      MPI_Comm_size(MPI_COMM_WORLD, &n_process);

      std::vector<unsigned long> processes_work_sizes(n_process);
      MPI_Allgather(&local_work_size, 1, MPI_UNSIGNED_LONG, processes_work_sizes.data(), 1,
                    MPI_UNSIGNED_LONG, MPI_COMM_WORLD);

      return createSubCommunicators(processes_work_sizes);
    }
  }   // anonymous namespace


  void MPIParallelScheduler::run() {
    int rank = 0, world_n_process = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_n_process);

    // TODO: Refactor me!
    epochStart();
    auto job = ParallelScheduler::getJob();
    size_t global_work_size = job.getGlobalWorkSize();
    size_t batch_size = job.getBatchSize();

    BatchProgression progression(job.getInputs(), job.getTargets());

    auto sub_comms = synchronizeGlobalWorkSize(getJob().getGlobalWorkSize());
    size_t current_comm_index = 0;
    auto mpi_op = (MPIMLPOptimizer::Operation *) ParallelScheduler::optimizer_operation.get();
    for (size_t current_size = 0; current_size < global_work_size; current_size += batch_size) {
      if (current_size > sub_comms.at(current_comm_index).first) {
        current_comm_index++;
        assert(current_comm_index < sub_comms.size());
        mpi_op->setCommunicator(sub_comms.at(current_comm_index).second);
        tscl::logger("[P" + std::to_string(rank) + "]: " + "Switching to sub-communicator " +
                             std::to_string(current_comm_index),
                     tscl::Log::Warning);
      }

      size_t current_batch_size = std::min(global_work_size - current_size, batch_size);
      ParallelScheduler::batch_dispatcher->dispatch(progression, current_batch_size,
                                                    *ParallelScheduler::optimizer_operation);

      ParallelScheduler::updateModel();
    }
    tscl::logger("[P" + std::to_string(rank) + "]: " + "MPIParallelScheduler::run: Finished",
                 tscl::Log::Warning);

    ParallelScheduler::optimizer->update();
    ParallelScheduler::endEpoch();
  }


  std::unique_ptr<ParallelScheduler> MPIParallelScheduler::Builder::build() const {
    if (not optimizer or devices.empty() or not job.isValid()) {
      throw std::runtime_error(
              "MPIParallelScheduler::Builder: not all required parameters are set");
    }

    ParallelScheduler::Policy policy(max_thread, multiple_thread_per_device, devices);
    auto scheduler = ParallelScheduler::makeWithDefaultDispatcher(job, *optimizer, policy);
    return std::make_unique<MPIParallelScheduler>(std::move(scheduler));
  }
}   // namespace nnet