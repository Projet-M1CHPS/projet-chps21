#include "MPIParallelScheduler.hpp"
#include "MPIMLPOptimizer.hpp"
#include "math/clFTensor.hpp"
#include <boost/asio.hpp>
#include <boost/asio/thread_pool.hpp>
#include <future>

using namespace math;
using namespace boost;

namespace nnet {
  std::string MPI_Comm_toString(MPI_Comm &comm) {
    int comm_size;
    MPI_Comm_size(comm, &comm_size);

    MPI_Group world_group;
    MPI_Comm_group(comm, &world_group);
    MPI_Group comm_group;
    MPI_Comm_group(comm, &comm_group);
    std::vector<int> world_ranks(comm_size);
    std::vector<int> comm_ranks(comm_size);
    MPI_Group_translate_ranks(world_group, comm_size, world_ranks.data(), comm_group,
                              comm_ranks.data());
    std::stringstream ss;
    for (int i = 0; i < comm_size; i++) { ss << comm_ranks[i] << ((i < comm_size - 1) ? "," : ""); }
    return "[" + ss.str() + "]";
  }

  /**
   * @param ranks_to_keep
   * @return the new MPI_Comm
   * @warning returned MPI_Comm can be MPI_COMM_NULL
   */
  MPI_Comm newCommWith(const std::vector<int> &ranks_to_keep) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::stringstream ss;
    for (int i = 0; i < ranks_to_keep.size(); i++)
      ss << ranks_to_keep[i] << ((i < ranks_to_keep.size() - 1) ? "," : "");
    tscl::logger("[P" + std::to_string(rank) + "]: " + "Keeping ranks: " + "[P" +
                 std::to_string(rank) + "]: " + "::newCommWith(" + ss.str() + ")");

    MPI_Comm new_comm;
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group new_group;
    MPI_Group_incl(world_group, (int) ranks_to_keep.size(), ranks_to_keep.data(), &new_group);
    assert(new_group != MPI_GROUP_NULL);

    MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);
    return new_comm;
  }

  // print every rank in the communicator and its rank in the world communicator
  void rankInWorld(MPI_Comm &comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int worldRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    std::cout << "rank " << rank << " in world " << worldRank << " of " << size << " ranks"
              << std::endl;
  }

  MPI_Comm newCommWithout(const MPI_Comm &initial_comm, int rank_to_remove) {
    tscl::logger("::newCommWithout(..., " + std::to_string(rank_to_remove) + ")",
                 tscl::Log::Warning);
    int initial_comm_size, r;
    MPI_Comm_size(initial_comm, &initial_comm_size);
    if (initial_comm_size <= 2) return initial_comm;
    assert(rank_to_remove >= 0);
    assert(rank_to_remove < initial_comm_size);

    std::vector<int> ranks;
    for (r = 0; r < rank_to_remove; ++r) ranks.push_back(r);
    for (r = rank_to_remove + 1; r < initial_comm_size; ++r) ranks.push_back(r);

    std::string msg("MPI_Comm_split keep: ");
    for (auto cur_rank : ranks) msg += std::to_string(cur_rank) + " ";
    tscl::logger(msg, tscl::Log::Warning);

    MPI_Group initial_group, new_group;
    MPI_Comm_group(initial_comm, &initial_group);
    MPI_Group_incl(initial_group, initial_comm_size - 1, ranks.data(), &new_group);
    assert(new_group != MPI_GROUP_NULL);

    MPI_Comm new_comm;
    MPI_Comm_create(initial_comm, new_group, &new_comm);
    assert(new_comm != MPI_COMM_NULL);

    return new_comm;
  }

  std::vector<MPI_Comm> MPIParallelScheduler::synchronizeGlobalWorkSize() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    tscl::logger("[P" + std::to_string(rank) +
                 "]: " + "MPIParallelScheduler::synchronizeGlobalWorkSize");
    int n_process = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n_process);

    processes_work_sizes.resize(n_process);

    auto work_size = (unsigned long) getJob().getGlobalWorkSize();
    MPI_Allgather(&work_size, 1, MPI_UNSIGNED_LONG, processes_work_sizes.data(), 1,
                  MPI_UNSIGNED_LONG, MPI_COMM_WORLD);

    // print process work sizes
    std::cout << "[P" + std::to_string(rank) + "]: " + "processes work sizes: ";
    for (auto &size : processes_work_sizes) std::cout << size << " ";
    std::cout << std::endl;

    // Todo: Refactor this, poorly thought out algorithm
    std::vector<MPI_Comm> new_comms;
    new_comms.push_back(MPI_COMM_WORLD);
    size_t current_work_size = processes_work_sizes[0];
    for (int i = 0; i < n_process - 1; ++i) {
      std::vector<int> ranks_in_new_comm;
      for (int j = 1; j < n_process; j++)
        if (processes_work_sizes[j] >= current_work_size) ranks_in_new_comm.push_back(j);

      current_work_size = processes_work_sizes[i];
      auto new_comm = newCommWith(ranks_in_new_comm);
      // Necessary because MPI does not allow to create twice the same communicator
      if (new_comm != MPI_COMM_NULL) new_comms.push_back(new_comm);
      else
        new_comms.emplace_back(new_comms.back());
    }

    tscl::logger("[P" + std::to_string(rank) +
                 "]: " + "MPIParallelScheduler::synchronizeGlobalWorkSize done: " +
                 std::to_string(new_comms.size()) + " communicators");
    return new_comms;
  }

  void MPIParallelScheduler::run() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::cout << "[P" + std::to_string(rank) + "]: " + "MPIParallelScheduler::run: Starting"
              << std::endl;
    int world_n_process = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_n_process);

    // TODO: Refactor me!
    epochStart();
    auto job = ParallelScheduler::getJob();
    size_t global_work_size = job.getGlobalWorkSize();
    size_t batch_size = job.getBatchSize();
    std::cout << "[P" + std::to_string(rank) + "]: " + "Global work size: " << global_work_size
              << std::endl;

    BatchProgression progression(job.getInputs(), job.getTargets());

    tscl::logger("[P" + std::to_string(rank) +
                         "]: " + "global_work_size: " + std::to_string(global_work_size),
                 tscl::Log::Warning);

    auto comms = synchronizeGlobalWorkSize();
    size_t current_comm_index = 0;
    size_t current_process_work_index = 0;
    MPI_Comm current_comm = comms[current_comm_index];
    for (size_t current_size = 0; current_size < global_work_size; current_size += batch_size) {
      if (current_size > processes_work_sizes.at(current_process_work_index)) {
        current_process_work_index++;
        current_comm_index++;
        assert(current_comm_index < comms.size());
        current_comm = comms.at(current_comm_index);
        tscl::logger("[P" + std::to_string(rank) +
                             "]: " + "MPIParallelScheduler::run: current_comm_index: " +
                             std::to_string(current_comm_index),
                     tscl::Log::Warning);
      }

      size_t current_batch_size = std::min(global_work_size - current_size, batch_size);
      ParallelScheduler::batch_dispatcher->dispatch(progression, current_batch_size,
                                                    *ParallelScheduler::optimizer_operation);

      auto op = (MPIMLPOptimizer::Operation *) ParallelScheduler::optimizer_operation.get();
      op->setCommunicator(current_comm);
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