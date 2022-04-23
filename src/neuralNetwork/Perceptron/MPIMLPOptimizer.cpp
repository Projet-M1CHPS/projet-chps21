#include "MPIMLPOptimizer.hpp"

using namespace math;

namespace nnet {
  namespace {
    std::vector<size_t>
    gatherContributions(const std::unique_ptr<MLPOptimizer::WeightUpdateCache> &send_cache,
                        MPI_Comm &comm) {
      int rank = 0, n_process = 0;
      MPI_Comm_rank(comm, &rank);
      MPI_Comm_size(comm, &n_process);

      std::vector<unsigned long> contributions;
      if (rank == 0) contributions.resize(n_process);
      auto send_contribution = (unsigned long) send_cache->getContribution();
      MPI_Gather(&send_contribution, 1, MPI_UNSIGNED_LONG, contributions.data(), 1,
                 MPI_UNSIGNED_LONG, 0, comm);

      return contributions;
    }

    std::vector<std::vector<math::clFMatrix>>
    gatherWeightUpdates(const std::unique_ptr<MLPOptimizer::WeightUpdateCache> &send_cache,
                        MPI_Comm &comm) {
      int rank = 0, n_process = 0;
      MPI_Comm_rank(comm, &rank);
      MPI_Comm_size(comm, &n_process);

      auto &send_weight_updates = send_cache->getWeightUpdates();
      auto rows = send_weight_updates.at(0).getRows(), cols = send_weight_updates.at(0).getCols();
      auto send_weight_updates_matrix_size = rows * cols;

      // Recv buffer for matrices
      std::vector<float> recv_weight_updates_buffer(n_process * send_weight_updates.size() *
                                                    send_weight_updates_matrix_size);

      // Gathering
      for (size_t i = 0; i < send_weight_updates.size(); ++i) {
        float *matrix_ptr = send_weight_updates.at(i).toFloatMatrix(true).getData();
        MPI_Gather(matrix_ptr, (int) send_weight_updates_matrix_size, MPI_FLOAT,
                   &recv_weight_updates_buffer.at(i * send_weight_updates_matrix_size),
                   (int) send_weight_updates_matrix_size, MPI_FLOAT, 0, comm);
      }

      // Recreate matrices from buffer
      std::vector<std::vector<math::clFMatrix>> recv_weight_updates;

      if (rank == 0) {
        recv_weight_updates.resize(n_process);
        for (size_t i = 0; i < n_process; ++i) {
          recv_weight_updates.at(i).resize(send_weight_updates.size());
          for (size_t j = 0; j < send_weight_updates.size(); ++j)
            recv_weight_updates.at(i).emplace_back(recv_weight_updates_buffer.data() +
                                                           i * send_weight_updates.size() +
                                                           (j * send_weight_updates_matrix_size),
                                                   rows, cols, true);
        }
      }
      return recv_weight_updates;
    }
  }   // namespace

  void MPIMLPOptimizer::Operation::reduceAll(cl::CommandQueue &queue) {
    // Shared between processes
    std::vector<WeightUpdateCache> recv_caches;
    size_t totalContributions = 0;

    // MPI environment
    int rank = 0, n_process = 0;
    MPI_Comm_rank(current_comm, &rank);
    MPI_Comm_size(current_comm, &n_process);

    // Process-local reduction
    for (size_t i = 1; i < caches.size(); i++) caches[0]->reduce(*caches[i], queue);

    // Gather the contributions from all processes
    auto global_contributions = gatherContributions(caches.at(0), current_comm);
    if (rank == 0) assert(global_contributions.size() == n_process);
    else
      assert(global_contributions.empty());

    // Sharing weight_updates attribute
    auto global_weight_updates = gatherWeightUpdates(caches.at(0), current_comm);
    if (rank == 0) assert(global_weight_updates.size() == n_process);
    else
      assert(global_weight_updates.empty());

    if (rank == 0) {
      // Recreate a vector of WeightUpdateCache
      for (size_t p = 0; p < n_process; p++) {
        std::vector<math::clFMatrix> local_weight_updates = std::move(global_weight_updates.at(p));
        WeightUpdateCache cache(std::move(local_weight_updates), global_contributions.at(p));
        recv_caches.push_back(std::move(cache));
      }

      // Reduce all received caches into zero-th cache
      for (size_t i = 1; i < recv_caches.size(); i++)
        recv_caches.at(0).reduce(recv_caches[i], queue);
      // Todo: Assign cache.at(0) to recv_caches.at(0)

      // Average the contributions
      totalContributions = 0;
      for (auto &cache : recv_caches) totalContributions += cache.getContribution();
      caches.at(0)->setContribution(totalContributions / n_process);
    }
  }


  void MPIMLPOptimizer::Operation::applyChanges(cl::CommandQueue &queue) {
    MLPOptimizer::Operation::applyChanges(queue);
    synchronizeModel();
  }

  void MPIMLPOptimizer::Operation::clearChanges(cl::CommandQueue &queue) {
    MLPOptimizer::Operation::clearChanges(queue);
  }

  std::unique_ptr<Optimizer::Operation> MPIMLPOptimizer::makeOperationImpl() {
    return std::make_unique<Operation>(*this);
  }


  /**
   * @before caches.at(0)->contribution is the mean contribution of all processes
   * @brief Broadcast the contribution & 0-th cache matrices to all processes
   */
  void MPIMLPOptimizer::Operation::synchronizeModel() {
    int rank = 0;
    MPI_Comm_rank(current_comm, &rank);

    size_t mean_contribution = caches.at(0)->getContribution();
    MPI_Bcast(&mean_contribution, 1, MPI_UNSIGNED, 0, current_comm);

    // Update the contribution of 0-th cache
    caches.at(0)->setContribution(mean_contribution);

    // Share & Update weight_updates attribute
    cl::CommandQueue q = utils::cl_wrapper.getDefaultQueue();
    // Todo: Vector of pointers & send them one by one, asynchronously
    for (const auto &ith_updated_weight : caches.at(0)->getWeightUpdates()) {
      void *data_ptr = q.enqueueMapBuffer(
              ith_updated_weight.getBuffer(), CL_TRUE, rank == 0 ? CL_MAP_READ : CL_MAP_WRITE,
              ith_updated_weight.getOffsetInBytes(), ith_updated_weight.sizeInBytes());

      MPI_Bcast(data_ptr, (int) ith_updated_weight.size(), MPI_FLOAT, 0, current_comm);

      q.enqueueUnmapMemObject(ith_updated_weight.getBuffer(), data_ptr);
    }
    q.finish();
  }
}   // namespace nnet