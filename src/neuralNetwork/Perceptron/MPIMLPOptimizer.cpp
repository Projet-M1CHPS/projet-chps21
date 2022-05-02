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
      assert(n_process > 1);

      auto &send_weight_updates = send_cache->getWeightUpdates();

      // Gathering
      std::vector<std::vector<math::clFMatrix>> recv_weight_updates((rank == 0) ? n_process : 0);
      std::vector<float> recv_raw_matrices;
      cl::CommandQueue q = cl::CommandQueue::getDefault();
      for (auto &mat : send_weight_updates) {
        int rows = (int) mat.getRows(), cols = (int) mat.getCols();
        int matrix_size = (int) mat.size();
        recv_raw_matrices.resize((rank == 0) ? n_process * matrix_size : 0);

        // Gather raw matrices from all processes
        void *send_mat_ptr = q.enqueueMapBuffer(mat.getBuffer(), CL_TRUE, CL_MAP_READ, 0,
                                                matrix_size * sizeof(float));
        MPI_Gather(send_mat_ptr, matrix_size, MPI_FLOAT, recv_raw_matrices.data(), matrix_size,
                   MPI_FLOAT, 0, comm);
        if (rank > 0) q.enqueueUnmapMemObject(mat.getBuffer(), send_mat_ptr);


        if (rank == 0) {
          for (size_t p = 0; p < n_process; p++)
            recv_weight_updates[p].emplace_back(recv_raw_matrices.data() + p * matrix_size, rows,
                                                cols, true);

          recv_raw_matrices.clear();
        }
      }
      q.finish();
      return recv_weight_updates;
    }
  }   // namespace

  void MPIMLPOptimizer::Operation::reduceAll(cl::CommandQueue &queue) {
    // Shared between processes
    std::vector<WeightUpdateCache> recv_caches;

    // MPI environment
    int rank = 0, n_process = 0;
    MPI_Comm_rank(current_comm, &rank);
    MPI_Comm_size(current_comm, &n_process);

    // Process-local reduction
    for (size_t i = 1; i < caches.size(); i++) caches[0]->reduce(*caches[i], queue);

    // If there is only one process, no synchronization is needed
    if (n_process == 1) return;

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
      for (size_t p = 0; p < n_process; p++)
        recv_caches.emplace_back(*optimizer, std::move(global_weight_updates[p]),
                                 global_contributions[p]);

      // Reduce all received caches into zero-th cache
      for (size_t i = 1; i < recv_caches.size(); i++) recv_caches[0].reduce(recv_caches[i], queue);

      // Assign: cache.at(0) <-- recv_caches.at(0)
      *caches[0] = std::move(recv_caches.at(0));
    }
  }


  void MPIMLPOptimizer::Operation::applyChanges(cl::CommandQueue &queue) {
    MLPOptimizer::Operation::applyChanges(queue);
    queue.finish();
    synchronizeModel();
  }

  void MPIMLPOptimizer::Operation::clearChanges(cl::CommandQueue &queue) {
    MLPOptimizer::Operation::clearChanges(queue);
  }

  std::unique_ptr<Optimizer::Operation> MPIMLPOptimizer::makeOperationImpl() {
    return std::make_unique<MPIMLPOptimizer::Operation>(*this);
  }


  /**
   * @before caches.at(0)->contribution is the mean contribution of all processes
   * @brief Broadcast the contribution & 0-th cache matrices to all processes
   */
  void MPIMLPOptimizer::Operation::synchronizeModel() {
    int rank = 0, n_process = 0;
    MPI_Comm_rank(current_comm, &rank);
    MPI_Comm_size(current_comm, &n_process);

    if (n_process == 1) return;

    auto &weights = optimizer->getNeuralNetwork()->getWeights();
    // Map asynchronous buffer for weight_updates
    std::vector<void *> data_ptrs(weights.size());
    data_ptrs.reserve(weights.size());
    for (size_t i = 0; i < weights.size(); i++)
      data_ptrs[i] = enqueueMapBuffer(weights[i].getBuffer(), CL_FALSE,
                                      rank == 0 ? CL_MAP_READ : CL_MAP_WRITE,
                                      weights[i].getOffsetInBytes(), weights[i].sizeInBytes());
    utils::cl_wrapper.getDefaultQueue().finish();

    // Broadcast weight_updates
    std::vector<MPI_Request> requests(weights.size());
    for (size_t i = 0; i < weights.size(); i++)
      MPI_Ibcast(data_ptrs[i], (int) weights[i].size(), MPI_FLOAT, 0, current_comm, &requests[i]);
    MPI_Waitall((int) requests.size(), requests.data(), MPI_STATUSES_IGNORE);

    // Unmap buffers
    for (size_t i = 0; i < weights.size(); i++)
      enqueueUnmapMemObject(weights[i].getBuffer(), data_ptrs[i]);
    utils::cl_wrapper.getDefaultQueue().finish();
  }

  MPI_Comm MPIMLPOptimizer::Operation::getCommunicator() { return current_comm; }
}   // namespace nnet
